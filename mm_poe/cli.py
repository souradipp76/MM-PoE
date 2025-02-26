# a framework for inference on multiple choice tasks.
from argparse import Namespace
import copy
import logging
import os
import subprocess
import pathlib

import questionary
import torch
from torch.utils.data import DataLoader

from mm_poe.methods.utils.data import (
    create_multiple_choice_prompt,
    preprocess_function_seq2seq_vqa,
    preprocess_function_seq2seq_vqa_channel,
    preprocess_function_causal_vqa,
    preprocess_function_causal_vqa_channel,
)
from mm_poe.methods.utils.methods import (
    compute_conditional_score_seq2seq_vqa,
    compute_conditional_score_causal_vqa,
    compute_mask_process_of_elimination,
    inference_process_of_elimination,
    inference_language_modeling,
    inference_calibration,
)
from mm_poe.methods.utils.utils import load_data, load_model, set_seed

all_checkpoints = {
    "BLIP2-OPT": ["Salesforce/blip2-opt-2.7b"],
    "BLIP2-T5": ["Salesforce/blip2-flan-t5-xl"],
    "InstructBLIP": ["Salesforce/instructblip-vicuna-7b"],
    "GIT": ["microsoft/git-base-vqav2", "microsoft/git-base-textvqa"],
    "PaliGemma": [
        "google/paligemma-3b-ft-science-qa-448",
        "google/paligemma-3b-ft-vqav2-448",
        "google/paligemma-3b-ft-ai2d-448",
    ],
    "Idefics2": ["HuggingFaceM4/idefics2-8b"],
}

logger = logging.getLogger(__name__)


def main():
    """
    The main function executes on commands:
    `python -m mm_poe` and `$ mm_poe `.
    """
    # step 1: collect arguments
    args = Namespace()
    args.seed = 0

    args.model_family = questionary.select(
        message="Select model family?",
        choices=[
            "BLIP2-OPT",
            "BLIP2-T5",
            "InstructBLIP",
            "GIT",
            "PaliGemma",
            "Idefics2",
        ],
        default="GIT",
    ).ask()

    checkpoints_choices = all_checkpoints[args.model_family]
    args.checkpoint = questionary.select(
        message="Select model checkpoint?",
        choices=checkpoints_choices,
        default=checkpoints_choices[0],
    ).ask()

    args.loading_precision = questionary.select(
        message="Select model precision?",
        choices=["FP32", "FP16", "BF16", "INT8", "INT4"],
        default="FP32",
    ).ask()

    args.output_dir = questionary.path(
        message="Model output directory?",
        only_directories=True,
        default="./models/",
    ).ask()

    args.dataset = "single_inference"
    args.batch_size = 1
    args.sample = 1
    args.n_shot = 0

    args.multiple_choice_prompt = ""
    args.calibration_prompt = " the answer is:"
    args.process_of_elimination_prompt = (
        "Select the most suitable option "
        + "to answer the question. Ignore [MASK] options."
    )

    args.scoring_method_for_process_of_elimination = questionary.select(
        message="Select scoring method?",
        choices=[
            "channel",
            "calibration",
            "language_modeling",
            "multiple_choice_prompt",
        ],
        default="language_modeling",
    ).ask()

    args.mask_strategy_for_process_of_elimination = questionary.select(
        message="Select mask strategy?",
        choices=["below_average", "lowest"],
        default="below_average",
    ).ask()

    args.prompting_method_for_process_of_elimination = "multiple_choice_prompt"
    args.mask_token = None

    args.question = questionary.text("Question:").ask().strip()
    args.choices = questionary.text("Choices [comma seprated]:").ask()
    args.choices = [choice.strip() for choice in args.choices.split(",")]
    args.num_options = len(args.choices)
    args.image_path = questionary.path(
        "Image Path?", default="./images/image.png"
    ).ask()
    args.label = questionary.select(
        message="Ground Truth Option:",
        choices=[str(x) for x in range(args.num_options)],
    ).ask()
    args.label = int(args.label)
    args.method = "process_of_elimination"

    # print(args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # step 2: set random seed to ensure reproducibility.
    logger.info(f"Set random seed to {args.seed}.")
    set_seed(args.seed)

    # step 3: download model
    logger.info(f"Download {args.model_family} model: {args.checkpoint}.")
    model_downloader_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        "models/model_downloaders/model_downloaders.py",
    )
    subprocess.call(
        f"python {model_downloader_path} "
        + f"--model_family {args.model_family} "
        + f"--checkpoint {args.checkpoint} "
        + f"--output_dir {args.output_dir}",
        shell=True,
    )

    # step 4: load model, tokenizer.
    # Then move to gpu, and set to evaluation mode.
    logger.info(f"Load {args.model_family} model: {args.checkpoint}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    model_path = os.path.join(
        args.output_dir, args.model_family, args.checkpoint
    )
    model, tokenizer = load_model(device, model_path, args)
    if args.model_family in [
        "BLIP2-T5",
        "InstructBLIP",
        "PaliGemma",
        "Idefics2",
    ]:
        compute_func = compute_conditional_score_seq2seq_vqa
        preprocess_func = preprocess_function_seq2seq_vqa
        preprocess_func_channel = preprocess_function_seq2seq_vqa_channel
        remove_columns = [
            "header_input_ids",
            "header_attention_mask",
            "ending_input_ids",
            "ending_attention_mask",
            "images",
        ]
        processor = tokenizer
        tokenizer = processor.tokenizer
    elif args.model_family in ["BLIP2-OPT", "GIT"]:
        compute_func = compute_conditional_score_causal_vqa
        preprocess_func = preprocess_function_causal_vqa
        preprocess_func_channel = preprocess_function_causal_vqa_channel
        remove_columns = [
            "input_ids",
            "labels",
            "images",
            "ending_attention_mask",
        ]
        processor = tokenizer
        tokenizer = processor.tokenizer
    else:
        raise NotImplementedError

    # step 5: load and preprocess data.
    logger.info(f"Load data: {args.dataset}.")

    # evaluate on dataset
    multiple_choice_prompt = args.multiple_choice_prompt
    args.multiple_choice_prompt = None
    (
        ending_names,
        header_name,
        image_header_name,
        raw_dataset,
        n_shot_dataset,
    ) = load_data(args)

    mcp_args = copy.deepcopy(args)
    mcp_args.multiple_choice_prompt = multiple_choice_prompt
    _, _, _, raw_mcp_dataset, n_shot_mcp_dataset = load_data(mcp_args)

    logger.info(f"Preprocess data: {args.dataset}.")
    fn_kwargs = {
        "ending_names": ending_names,
        "header_name": header_name,
        "tokenizer": tokenizer,
        "processor": processor,
        "image_header_name": image_header_name,
    }
    num_of_options = len(ending_names)
    tokenized_dataset = raw_dataset.map(
        preprocess_func,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=False
    )

    # step 5: (evaluation) inference on data, and compute accuracy.
    logger.info(
        f"Start inference (method: {args.method}) on {args.dataset} "
        + f"using {args.model_family} model: {args.checkpoint}."
    )
    scoring_method = args.scoring_method_for_process_of_elimination
    logger.info(f"Step 1: Computing masks. Scoring method: {scoring_method}.")
    if scoring_method == "channel":
        tokenized_channel_dataset = raw_dataset.map(
            preprocess_func_channel,
            fn_kwargs=fn_kwargs,
            batched=True,
            batch_size=args.batch_size,
        )
        eval_channel_dataloader = DataLoader(
            tokenized_channel_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        avg_log_probs, _, _, lm_predictions = inference_language_modeling(
            model,
            eval_channel_dataloader,
            device,
            compute_func,
            tokenizer.pad_token_id,
        )
    elif scoring_method == "calibration":
        fn_kwargs = {
            "ending_names": ending_names,
            "header_name": "uncond_premise",  # the difference is here
            "tokenizer": tokenizer,
            "processor": processor,
            "image_header_name": image_header_name,
        }
        tokenized_calibration_dataset = raw_dataset.map(
            preprocess_func,
            fn_kwargs=fn_kwargs,
            batched=True,
            batch_size=args.batch_size,
        )
        eval_calibration_dataloader = DataLoader(
            tokenized_calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        avg_log_probs, _, _, lm_predictions = inference_calibration(
            model,
            eval_dataloader,
            eval_calibration_dataloader,
            device,
            compute_func,
            tokenizer.pad_token_id,
        )
    elif scoring_method == "language_modeling":
        avg_log_probs, _, _, lm_predictions = inference_language_modeling(
            model,
            eval_dataloader,
            device,
            compute_func,
            tokenizer.pad_token_id,
        )
    elif scoring_method == "multiple_choice_prompt":
        # mcp_args = copy.deepcopy(args)
        # mcp_args.multiple_choice_prompt = multiple_choice_prompt
        # _, _, raw_mcp_dataset, n_shot_mcp_dataset = load_data(mcp_args)
        # raw_mcp_dataset, n_shot_mcp_dataset = create_n_shot_splits(
        #     raw_mcp_dataset,
        #     n_shot_mcp_dataset,
        #     args
        # )
        tokenized_dataset = raw_mcp_dataset.map(
            preprocess_func,
            fn_kwargs=fn_kwargs,
            batched=True,
            batch_size=args.batch_size,
        )
        eval_mcp_dataloader = DataLoader(
            tokenized_dataset, batch_size=args.batch_size, shuffle=False
        )
        avg_log_probs, _, _, lm_predictions = inference_language_modeling(
            model,
            eval_mcp_dataloader,
            device,
            compute_func,
            tokenizer.pad_token_id,
        )
    else:
        raise NotImplementedError  # unlikely to happen.

    mask_strategy = args.mask_strategy_for_process_of_elimination
    if mask_strategy == "min_k":
        # masking the most k UNLIKELY options
        min_k = args.min_k
        if min_k >= num_of_options:
            min_k = num_of_options - 1
        mask_kwargs = {
            "min_k": min_k,
        }
    else:
        mask_kwargs = {}
    masks = compute_mask_process_of_elimination(
        avg_log_probs, mask_strategy, **mask_kwargs
    )
    # construct an oracle mask that only keeps the correct lable to 1,
    # and other options to 0
    # oracle_masks = torch.zeros_like(avg_log_probs)
    # oracle_masks[torch.arange(oracle_masks.size(0)), \
    #              tokenized_dataset["label"]] = 1
    masks = masks.to(torch.float32)
    # compute mask accuracy, i.e.,
    # check whether mask that correspond to labels is 1
    mask_result = masks[
        torch.arange(masks.size(0)), tokenized_dataset["label"]
    ]
    mask_accuracy = torch.sum(mask_result) / mask_result.size(0)
    logger.info(f"Mask accuracy: {mask_accuracy}")
    args.mask_accuracy = mask_accuracy.item()
    masked_dataset = tokenized_dataset.map(
        lambda example, idx: {"mask": masks[idx]},
        with_indices=True,
        batched=True,
        remove_columns=remove_columns,
    )

    prompting_method = args.prompting_method_for_process_of_elimination
    logger.info(
        "Step 2: Creating multiple choice prompt. "
        + f"Prompting method: {prompting_method}."
    )
    # if args.prompting_method_for_process_of_elimination
    # mcp_kwargs = {"multiple_choice_prompt": multiple_choice_prompt,}
    mask_token = args.mask_token
    if mask_token is not None:
        if mask_token == "":
            args.process_of_elimination_prompt = (
                args.process_of_elimination_prompt.replace("[MASK]", "empty")
            )
        else:
            args.process_of_elimination_prompt = (
                args.process_of_elimination_prompt.replace(
                    "[MASK]", mask_token
                )
            )
    mcp_kwargs = {
        "multiple_choice_prompt": args.process_of_elimination_prompt,
        "scoring_method": scoring_method,
        "num_of_options": num_of_options,
        "mask_token": mask_token,
    }
    mcp_dataset = masked_dataset.map(
        create_multiple_choice_prompt, fn_kwargs=mcp_kwargs
    )

    logger.info("Step 3: Final Inference")
    mcp_dataset = mcp_dataset.map(
        preprocess_func,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=args.batch_size,
    )
    eval_mcp_dataloader = DataLoader(
        mcp_dataset, batch_size=args.batch_size, shuffle=False
    )
    poe_avg_log_probs, lm_accuracy, _, lm_predictions = (
        inference_process_of_elimination(
            model,
            eval_mcp_dataloader,
            device,
            compute_func,
            tokenizer.pad_token_id,
        )
    )
    option = int(lm_predictions.numpy()[0])
    logger.info(f"Predicted Option: {option}. Answer: {args.choices[option]}")
