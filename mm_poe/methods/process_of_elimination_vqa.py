# a framework for inference on multiple choice tasks.
import copy
import logging
import os

import torch
from torch.utils.data import DataLoader


from utils.data import (
    upload_to_huggingface_hub,
    preprocess_function_seq2seq,
    preprocess_function_causal,
    preprocess_function_causal_channel,
    preprocess_function_seq2seq_channel,
    create_multiple_choice_prompt,
    generate_n_shot_poe_demonstrations,
    create_n_shot_splits,
    preprocess_function_seq2seq_vqa,
    preprocess_function_seq2seq_vqa_channel,
    preprocess_function_causal_vqa,
    preprocess_function_causal_vqa_channel,
)
from utils.methods import (
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
    compute_conditional_score_seq2seq_vqa,
    compute_conditional_score_causal_vqa,
    compute_mask_process_of_elimination,
    inference_process_of_elimination,
    inference_language_modeling,
    inference_calibration,
)
from utils.utils import (
    load_data,
    load_model,
    parse_args,
    set_seed,
    write_to_csv,
)

logger = logging.getLogger(__name__)


def main():
    # import pdb; pdb.set_trace()

    # step 1: argument parser, and logger
    args = parse_args()
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

    # step 3: load model, tokenizer.
    # Then move to gpu, and set to evaluation mode.
    logger.info(f"Load {args.model_family} model: {args.checkpoint}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get model path: ../models/args.model_family/args.checkpoint
    model_path = os.path.join("../models", args.model_family, args.checkpoint)
    model, tokenizer = load_model(device, model_path, args)
    if args.model_family in ["GPT2", "Pythia", "OPT-IML", "Dolly"]:
        compute_func = compute_conditional_score_causal
        preprocess_func = preprocess_function_causal
        preprocess_func_channel = preprocess_function_causal_channel
        remove_columns = ["input_ids", "labels", "ending_attention_mask"]
    elif args.model_family in ["T5", "FLAN-T5"]:
        compute_func = compute_conditional_score_seq2seq
        preprocess_func = preprocess_function_seq2seq
        preprocess_func_channel = preprocess_function_seq2seq_channel
        remove_columns = [
            "header_input_ids",
            "header_attention_mask",
            "ending_input_ids",
            "ending_attention_mask",
        ]
    elif args.model_family in [
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

    # step 4: load and preprocess data.
    args.datasets = args.datasets.split()
    logger.info(f"Load data: {args.datasets}.")

    # evaluate on each dataset
    multiple_choice_prompt = args.multiple_choice_prompt
    for dataset in args.datasets:
        args.dataset = dataset
        # multiple_choice_prompt = args.multiple_choice_prompt
        args.multiple_choice_prompt = None
        if args.dataset in ["vqa", "scienceqa", "ai2d", "custom_dataset"]:
            (
                ending_names,
                header_name,
                image_header_name,
                raw_dataset,
                n_shot_dataset,
            ) = load_data(args)
        else:
            ending_names, header_name, raw_dataset, n_shot_dataset = load_data(
                args
            )
        raw_dataset, n_shot_dataset, n_shot_demonstrations = (
            create_n_shot_splits(raw_dataset, n_shot_dataset, args)
        )

        mcp_args = copy.deepcopy(args)
        mcp_args.multiple_choice_prompt = multiple_choice_prompt
        if args.dataset in ["vqa", "scienceqa", "ai2d", "custom_dataset"]:
            _, _, _, raw_mcp_dataset, n_shot_mcp_dataset = load_data(mcp_args)
        else:
            _, _, raw_mcp_dataset, n_shot_mcp_dataset = load_data(mcp_args)
        raw_mcp_dataset, n_shot_mcp_dataset, _ = create_n_shot_splits(
            raw_mcp_dataset, n_shot_mcp_dataset, args
        )

        logger.info(f"Preprocess data: {args.dataset}.")
        if args.model_family in [
            "BLIP2-OPT",
            "BLIP2-T5",
            "InstructBLIP",
            "GIT",
            "PaliGemma",
            "Idefics2",
        ]:
            fn_kwargs = {
                "ending_names": ending_names,
                "header_name": header_name,
                "tokenizer": tokenizer,
                "processor": processor,
                "image_header_name": image_header_name,
                "image_token": "<image>",
            }
            if args.model_family in ["GIT"]:
                fn_kwargs["image_token"] = ""
        else:
            fn_kwargs = {
                "ending_names": ending_names,
                "header_name": header_name,
                "tokenizer": tokenizer,
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
        logger.info(
            f"Step 1: Computing masks. Scoring method: {scoring_method}."
        )
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
            avg_log_probs, _, _, _ = inference_language_modeling(
                model,
                eval_channel_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        elif scoring_method == "calibration":
            if args.model_family in [
                "BLIP2-OPT",
                "BLIP2-T5",
                "InstructBLIP",
                "GIT",
                "PaliGemma",
                "Idefics2",
            ]:
                fn_kwargs = {
                    "ending_names": ending_names,
                    "header_name": "uncond_premise",  # the difference is here
                    "tokenizer": tokenizer,
                    "processor": processor,
                    "image_header_name": image_header_name,
                    "image_token": "<image>",
                }
                if args.model_family in ["GIT"]:
                    fn_kwargs["image_token"] = ""
            else:
                fn_kwargs = {
                    "ending_names": ending_names,
                    "header_name": "uncond_premise",  # the difference is here
                    "tokenizer": tokenizer,
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
            avg_log_probs, _, _, _ = inference_calibration(
                model,
                eval_dataloader,
                eval_calibration_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        elif scoring_method == "language_modeling":
            avg_log_probs, _, _, _ = inference_language_modeling(
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
            # raw_mcp_dataset, n_shot_mcp_dataset, args)
            tokenized_dataset = raw_mcp_dataset.map(
                preprocess_func,
                fn_kwargs=fn_kwargs,
                batched=True,
                batch_size=args.batch_size,
            )
            eval_mcp_dataloader = DataLoader(
                tokenized_dataset, batch_size=args.batch_size, shuffle=False
            )
            avg_log_probs, _, _, _ = inference_language_modeling(
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
        # tokenized_dataset["label"]] = 1
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
                    args.process_of_elimination_prompt.replace(
                        "[MASK]", "empty"
                    )
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
        # change n_shot format.
        if args.n_shot > 0:
            if (
                args.scoring_method_for_process_of_elimination
                == "multiple_choice_prompt"
            ):
                n_shot_demonstrations, n_shot_poe_demonstrations = (
                    generate_n_shot_poe_demonstrations(
                        n_shot_mcp_dataset, num_of_options
                    )
                )
            else:
                _, n_shot_poe_demonstrations = (
                    generate_n_shot_poe_demonstrations(
                        n_shot_mcp_dataset, num_of_options
                    )
                )
            mcp_dataset = mcp_dataset.map(
                lambda x: {
                    "premise": x["premise"].replace(
                        n_shot_demonstrations, n_shot_poe_demonstrations
                    )
                }
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
        poe_avg_log_probs, lm_accuracy, _, _ = (
            inference_process_of_elimination(
                model,
                eval_mcp_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        )

        # step 6: some postprocessing, including saving and displyaing output.
        save_path = os.path.join("../results", f"{args.method}.csv")
        logger.info(f"Save results to {save_path}.")
        save_args = copy.deepcopy(args)
        if mask_strategy == "min_k":
            save_args.mask_strategy_for_process_of_elimination = (
                f"min_k_{min_k}"
            )
        write_to_csv(save_path, save_args, lm_accuracy)

        # step 7: push data to HuggingFace Hub.
        if args.push_data_to_hub:
            logger.info(f"Push {args.dataset} to HuggingFace Hub.")
            # save the mcp dataset, which will be used by LLM.
            upload_to_huggingface_hub(mcp_dataset, args)


if __name__ == "__main__":
    main()
