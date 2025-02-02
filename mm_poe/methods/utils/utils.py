import argparse
import csv
import os
import random

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)
from datasets import Dataset

# import data.py, which is located in the same directory

from .data import (
    copa_loader,
    cqa_loader,
    obqa_loader,
    piqa_loader,
    qasc_loader,
    siqa_loader,
    winogrande_loader,
    date_understanding_loader,
    anli_loader,
    vqa_loader,
    scienceqa_loader,
    ai2d_loader,
    single_inference_loader,
    custom_loader,
)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser("Inference on multiple choice benchmarks")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=[
            "GPT2",
            "T5",
            "FLAN-T5",
            "Pythia",
            "OPT-IML",
            "Dolly",
            "BLIP2-OPT",
            "BLIP2-T5",
            "InstructBLIP",
            "GIT",
            "PaliGemma",
            "Idefics2",
        ],
        default=None,
        required=True,
        help="The moddel family, as checkpoints under the same "
        + "model family use same codes to download.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        required=True,
        help="The checkpoint name under a model family, "
        + "e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl.",
    )
    parser.add_argument(
        "--amateur_checkpoint",
        type=str,
        default=None,
        help="The amateur checkpoint name under a model family. "
        + "For constrative decoding.",
    )
    parser.add_argument(
        "--expert_method",
        type=str,
        choices=[
            "language_modeling",
            "calibration",
            "channel",
            "multiple_choice_prompt",
        ],
        default="language_modeling",
        help="The expert method. For constrative decoding.",
    )
    parser.add_argument(
        "--amateur_method",
        type=str,
        choices=[
            "language_modeling",
            "calibration",
            "channel",
            "multiple_choice_prompt",
        ],
        default="language_modeling",
        help="The amateur method. For constrative decoding.",
    )
    parser.add_argument(
        "--weighting_parameter",
        type=float,
        default=-1.0,
        help="The weighting parameter for constrative decoding. "
        + "It is applied to the amateur model.",
    )
    parser.add_argument(
        "--weighting_parameters",
        type=str,
        default=None,
        help="The weighting parameters for constrative decoding. "
        + "One weight for one dataset.",
    )
    parser.add_argument(
        "--num_random_search",
        type=int,
        default=0,
        help="The number of random search for the "
        + "weighting parameter for constrative decoding.",
    )
    parser.add_argument(
        "--loading_precision",
        type=str,
        choices=["FP32", "FP16", "BF16", "INT8", "INT4"],
        default="FP32",
        help="The precision of the model to be loaded.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        # choices=["copa", "cqa", "winogrande"],
        default=None,
        required=True,
        help="The datasets to inference on. "
        + "Pass multiple datasets separate by space",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="The number of samples to inference on. "
        + "If None, inference on the whole dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
        help="Number of few-shot demonstrations. 0 means zero-shot.",
    )
    parser.add_argument(
        "--multiple_choice_prompt",
        type=str,
        default=None,
        help="The multiple choice prompt.",
    )
    parser.add_argument(
        "--calibration_prompt",
        type=str,
        default=None,
        help="Calibration prompt, as in P(y|x)/P(y|calibration_prompt).",
    )
    parser.add_argument(
        "--do_channel",
        action="store_true",
        help="Whether to do channel, i.e., P(x|y_i).",
    )
    parser.add_argument(
        "--process_of_elimination_prompt",
        type=str,
        default=None,
        help="The process of elimination prompt. "
        + "It asks the model to ignore masked options.",
    )
    parser.add_argument(
        "--scoring_method_for_process_of_elimination",
        type=str,
        choices=[
            "language_modeling",
            "calibration",
            "channel",
            "multiple_choice_prompt",
        ],
        default="language_modeling",
        help="The scoring method for process of elimination.",
    )
    parser.add_argument(
        "--prompting_method_for_process_of_elimination",
        type=str,
        choices=["multiple_choice_prompt"],
        default="multiple_choice_prompt",
        help="The prompting method for process of elimination.",
    )
    parser.add_argument(
        "--mask_strategy_for_process_of_elimination",
        type=str,
        choices=["lowest", "below_average", "min_k"],
        default="lowest",
        help="The mask strategy for process of elimination.",
    )
    parser.add_argument(
        "--do_synonym",
        action="store_true",
        help="Whether to generate synonyms for options.",
    )
    parser.add_argument(
        "--number_of_synonyms",
        type=int,
        default=5,
        help="The number of synonyms to be used in the generative method.",
    )
    parser.add_argument(
        "--generate_synonyms_prompt",
        type=str,
        default=None,
        help="The prompt template for generating synonyms. "
        + "'option is replaced with actual options'",
    )
    parser.add_argument(
        "--push_data_to_hub",
        action="store_true",
        help="Whether to push the data to Hugging Face Hub. "
        + "This is convienient for LLM experiments.",
    )
    parser.add_argument(
        "--min_k",
        type=int,
        help="A masking strategy that masks the k most unlikely options",
    )
    parser.add_argument(
        "--mask_token",
        type=str,
        default=None,
        help="The mask token. If None, "
        + "use the default mask token of the model.",
    )

    args = parser.parse_args()
    return args


def load_data(args):
    # load test data for final performance.
    # load dev data to tune hyperparameters.
    # commonsense reasoning datasets
    train_file_path = None
    if args.dataset == "copa":
        ending_names = ["hypothesis0", "hypothesis1"]
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "copa-test.xml")
        train_file_path = os.path.join("../data", args.dataset, "copa-dev.xml")
        loader = copa_loader
    elif args.dataset == "cqa":
        ending_names = [
            "hypothesis0",
            "hypothesis1",
            "hypothesis2",
            "hypothesis3",
            "hypothesis4",
        ]
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        train_file_path = os.path.join("../data", args.dataset, "train.jsonl")
        loader = cqa_loader
    elif args.dataset == "obqa":
        ending_names = [
            "hypothesis0",
            "hypothesis1",
            "hypothesis2",
            "hypothesis3",
        ]
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "test.jsonl")
        train_file_path = os.path.join("../data", args.dataset, "train.jsonl")
        loader = obqa_loader
    elif args.dataset == "piqa":
        ending_names = ["hypothesis0", "hypothesis1"]
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "valid.jsonl")
        label_path = os.path.join("../data", args.dataset, "valid-labels.lst")
        file_path = [data_path, label_path]
        train_file_path = [
            path.replace("valid", "train") for path in file_path
        ]
        loader = piqa_loader
    elif args.dataset == "qasc":
        ending_names = [
            "hypothesis0",
            "hypothesis1",
            "hypothesis2",
            "hypothesis3",
            "hypothesis4",
            "hypothesis5",
            "hypothesis6",
            "hypothesis7",
        ]
        header_name = "premise"
        file_path = os.path.join("../data", args.dataset, "dev.jsonl")
        train_file_path = os.path.join("../data", args.dataset, "train.jsonl")
        loader = qasc_loader
    elif args.dataset == "siqa":
        ending_names = ["hypothesis0", "hypothesis1", "hypothesis2"]
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "dev.jsonl")
        label_path = os.path.join("../data", args.dataset, "dev-labels.lst")
        file_path = [data_path, label_path]
        train_file_path = [path.replace("dev", "train") for path in file_path]
        loader = siqa_loader
    elif args.dataset == "winogrande":
        ending_names = ["hypothesis0", "hypothesis1"]
        header_name = "premise"
        data_path = os.path.join("../data", args.dataset, "dev.jsonl")
        label_path = os.path.join("../data", args.dataset, "dev-labels.lst")
        file_path = [data_path, label_path]
        train_file_path = [
            path.replace("dev", "train_xs") for path in file_path
        ]
        loader = winogrande_loader
    # BIG-Bench tasks
    elif args.dataset == "disambiguation_qa":
        args.num_options = 3
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset == "conceptual_combinations":
        args.num_options = 4
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = []
        file_suffixes = [
            "contradictions",
            "emergent_properties",
            "fanciful_fictional_combinations",
            "homonyms",
            "invented_words",
            "surprising_uncommon_combinations",
        ]
        for suffix in file_suffixes:
            file_path.append(
                os.path.join(
                    "../data", "big_bench", f"{args.dataset}_{suffix}.json"
                )
            )
        loader = date_understanding_loader
    elif args.dataset == "date_understanding":
        args.num_options = 6
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset in [
        "emoji_movie",
        "evaluating_information_essentiality",
        "logical_args",
        "riddle_sense",
    ]:
        args.num_options = 5
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset in [
        "ruin_names",
        "temporal_sequences",
        "code_line_description",
        "crass_ai",
        "identify_math_theorems",
        "identify_odd_metaphor",
    ]:
        args.num_options = 4
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset == "penguins_in_a_table":
        # use the five object subtask
        args.num_options = 5
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset == "strange_stories":
        args.num_options = 4
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join(
                "../data", "big_bench", f"{args.dataset}_multiple_choice.json"
            )
        ]
        loader = date_understanding_loader
    elif args.dataset == "reasoning_about_colored_objects":
        args.num_options = 18
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    elif args.dataset == "symbol_interpretation":
        args.num_options = 5
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = []
        file_suffixes = [
            "adversarial",
            "emoji_agnostic",
            "name_agnostic",
            "plain",
            "tricky",
        ]
        for suffix in file_suffixes:
            file_path.append(
                os.path.join(
                    "../data", "big_bench", f"{args.dataset}_{suffix}.json"
                )
            )
        loader = date_understanding_loader
    elif args.dataset == "tracking_shuffled_objects":
        # use the five object subtask
        args.num_options = 5
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_suffix = "five_objects"
        file_path = [
            os.path.join(
                "../data", "big_bench", f"{args.dataset}_{file_suffix}.json"
            )
        ]
        loader = date_understanding_loader
    elif args.dataset in [
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
    ]:
        if "three" in args.dataset:
            args.num_options = 3
        elif "five" in args.dataset:
            args.num_options = 5
        else:
            args.num_options = 7
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        file_path = [
            os.path.join("../data", "big_bench", f"{args.dataset}.json")
        ]
        loader = date_understanding_loader
    # other datasets
    elif args.dataset == "anli":
        ending_names = ["hypothesis0", "hypothesis1", "hypothesis2"]
        header_name = "premise"
        file_path = []
        file_prefixes = ["R1", "R2", "R3"]
        for prefix in file_prefixes:
            file_path.append(
                os.path.join(
                    "../data", f"{args.dataset}", f"{prefix}_dev.jsonl"
                )
            )
        train_file_path = [path.replace("dev", "train") for path in file_path]
        loader = anli_loader
    elif args.dataset in ["anli_r1", "anli_r2", "anli_r3"]:
        args.num_options = 3
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        # ending_names = ['hypothesis0', 'hypothesis1', 'hypothesis2']
        header_name = "premise"
        prefix = args.dataset.split("_")[-1]
        file_path = [
            os.path.join("../data", "anli", f"{prefix.capitalize()}_dev.jsonl")
        ]
        # file_path = []
        # # file_prefixes = ["R1", "R2", "R3"]
        # for prefix in file_prefixes:
        #     file_path.append(os.path.join("../data", f"{args.dataset}", \
        #                                   f"{prefix}_dev.jsonl"))
        train_file_path = [path.replace("dev", "train") for path in file_path]
        loader = anli_loader
    elif args.dataset == "vqa":
        args.num_options = 18
        args.split = "val"
        file_path = os.path.join("../data", args.dataset)
        train_file_path = os.path.join("../data", args.dataset)
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        image_header_name = "image_path"
        loader = vqa_loader
    elif args.dataset == "scienceqa":
        args.num_options = 4
        args.split = "val"
        file_path = os.path.join("../data", args.dataset)
        train_file_path = os.path.join("../data", args.dataset)
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        image_header_name = "image_path"
        loader = scienceqa_loader
    elif args.dataset == "ai2d":
        args.num_options = 4
        file_path = os.path.join("../data", args.dataset)
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        image_header_name = "image_path"
        loader = ai2d_loader
    elif args.dataset == "single_inference":
        file_path = args.image_path
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        image_header_name = "image_path"
        loader = single_inference_loader
    elif args.dataset == "custom_dataset":
        args.num_options = 4
        args.split = "val"
        file_path = os.path.join("../data", args.dataset)
        train_file_path = os.path.join("../data", args.dataset)
        ending_names = [f"hypothesis{i}" for i in range(args.num_options)]
        header_name = "premise"
        image_header_name = "image_path"
        loader = custom_loader
    else:
        print(f"{args.dataset}: downloader not implemented.")
        return

    dev_data = loader(file_path, args)
    dev_dataset = Dataset.from_list(dev_data).with_format("torch")
    if train_file_path is not None:
        args.split = "train"
        train_data = loader(train_file_path, args)
        train_dataset = Dataset.from_list(train_data).with_format("torch")
    else:  # BB tasks have no train set.
        train_dataset = dev_dataset
    if args.dataset in [
        "vqa",
        "scienceqa",
        "ai2d",
        "single_inference",
        "custom_dataset",
    ]:
        return (
            ending_names,
            header_name,
            image_header_name,
            dev_dataset,
            train_dataset,
        )
    return ending_names, header_name, dev_dataset, train_dataset


def load_model(device, model_path, args):
    if args.model_family in ["GPT2", "Pythia", "OPT-IML", "Dolly"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForCausalLM
    elif args.model_family in ["T5", "FLAN-T5"]:
        tokenizer_func = AutoTokenizer
        model_func = AutoModelForSeq2SeqLM
    elif args.model_family in [
        "BLIP2-OPT",
        "BLIP2-T5",
        "InstructBLIP",
        "GIT",
        "PaliGemma",
        "Idefics2",
    ]:
        tokenizer_func = AutoProcessor
        model_func = AutoModelForVision2Seq
    else:
        print(f"{args.model_family}: downloader not implemented.")
        return
    if args.model_family == "Dolly":
        tokenizer = tokenizer_func.from_pretrained(
            model_path, padding_side="left"
        )
    elif args.model_family == "Idefics2":
        tokenizer = tokenizer_func.from_pretrained(
            model_path, do_image_splitting=False
        )
    else:
        tokenizer = tokenizer_func.from_pretrained(model_path)
    if args.model_family in ["GPT2", "Pythia", "Dolly"]:
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model_family in ["PaliGemma", "GIT"]:
        tokenizer.tokenizer.padding_side = "left"

    # load with different precision
    if args.loading_precision == "FP16":
        model = model_func.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16
        )
    elif args.loading_precision == "BF16":
        model = model_func.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.bfloat16
        )
    elif args.loading_precision == "INT8":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=200.0
        )
        model = model_func.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            quantization_config=quantization_config,
        )
    elif args.loading_precision == "INT4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = model_func.from_pretrained(
            model_path,
            device_map=device,
            quantization_config=quantization_config,
        )
    else:  # FP32
        model = model_func.from_pretrained(model_path, device_map=device)
    print(
        f"Memory footprint: {model.get_memory_footprint() / 1024 **3:.2f} GB."
    )
    return model, tokenizer


def write_to_csv(save_path, args, total_accuracy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    csv_exists = os.path.isfile(save_path)
    with open(save_path, "a+", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if args.method == "process_of_elimination":
            if not csv_exists:
                csvwriter.writerow(
                    [
                        "model_family",
                        "checkpoint",
                        "loading_precision",
                        "dataset",
                        "batch_size",
                        "method",
                        "scoring_method",
                        "prompting_method",
                        "mask_strategy",
                        "mask_token",
                        "seed",
                        "n_shot",
                        "sample",
                        "mask_accuracy",
                        "accuracy",
                    ]
                )
            csvwriter.writerow(
                [
                    args.model_family,
                    args.checkpoint,
                    args.loading_precision,
                    args.dataset,
                    args.batch_size,
                    args.method,
                    args.scoring_method_for_process_of_elimination,
                    args.prompting_method_for_process_of_elimination,
                    args.mask_strategy_for_process_of_elimination,
                    args.mask_token,
                    args.seed,
                    int(args.n_shot),
                    args.sample,
                    f"{args.mask_accuracy:.4f}",
                    f"{total_accuracy:.4f}",
                ]
            )
        elif args.method == "contrastive_decoding":
            if not csv_exists:
                csvwriter.writerow(
                    [
                        "model_family",
                        "checkpoint",
                        "amateur_checkpoint",
                        "loading_precision",
                        "dataset",
                        "batch_size",
                        "method",
                        "expert_method",
                        "amateur_method",
                        "weighting_parameter",
                        "seed",
                        "n_shot",
                        "sample",
                        "expert_accuracy",
                        "amateur_accuracy",
                        "accuracy",
                    ]
                )
            csvwriter.writerow(
                [
                    args.model_family,
                    args.checkpoint,
                    args.amateur_checkpoint,
                    args.loading_precision,
                    args.dataset,
                    args.batch_size,
                    args.method,
                    args.expert_method,
                    args.amateur_method,
                    args.weighting_parameter,
                    args.seed,
                    int(args.n_shot),
                    args.sample,
                    f"{args.expert_accuracy:.4f}",
                    f"{args.amateur_accuracy:.4f}",
                    f"{total_accuracy:.4f}",
                ]
            )
        elif args.method == "generate_synonyms":
            if not csv_exists:
                csvwriter.writerow(
                    [
                        "model_family",
                        "checkpoint",
                        "loading_precision",
                        "dataset",
                        "batch_size",
                        "method",
                        "number_of_synonyms",
                        "seed",
                        "n_shot",
                        "sample",
                        "accuracy",
                    ]
                )
            csvwriter.writerow(
                [
                    args.model_family,
                    args.checkpoint,
                    args.loading_precision,
                    args.dataset,
                    args.batch_size,
                    args.method,
                    args.number_of_synonyms,
                    args.seed,
                    int(args.n_shot),
                    args.sample,
                    f"{total_accuracy:.4f}",
                ]
            )
        else:
            if not csv_exists:
                csvwriter.writerow(
                    [
                        "model_family",
                        "checkpoint",
                        "loading_precision",
                        "dataset",
                        "batch_size",
                        "method",
                        "seed",
                        "n_shot",
                        "sample",
                        "accuracy",
                    ]
                )
            csvwriter.writerow(
                [
                    args.model_family,
                    args.checkpoint,
                    args.loading_precision,
                    args.dataset,
                    args.batch_size,
                    args.method,
                    args.seed,
                    int(args.n_shot),
                    args.sample,
                    f"{total_accuracy:.4f}",
                ]
            )
