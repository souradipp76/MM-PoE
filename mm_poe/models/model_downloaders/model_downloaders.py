"""Model Downloaders"""

import argparse
import glob
import os
import shutil

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoModelForVision2Seq,
)

all_checkpoints = {
    "GPT2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "Pythia": [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9B",
        "EleutherAI/pythia-12b",
        "EleutherAI/pythia-70m-deduped",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m-deduped",
        "EleutherAI/pythia-1b-deduped",
        "EleutherAI/pythia-1.4b-deduped",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9B-deduped",
        "EleutherAI/pythia-12b-deduped",
    ],
    "OPT-IML": ["facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b"],
    "T5": ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    "FLAN-T5": [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ],
    "MPT": [
        "mosaicml/mpt-7b",
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-chat",
        "mosaicml/mpt-7b-storywriter",
    ],
    "Dolly": ["databricks/dolly-v2-7b"],
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


def parse_args():
    """Parse Arguments"""
    parser = argparse.ArgumentParser(description="Language model downloaders.")

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
        help="The model family, as checkpoints under "
        + "the same model family use same codes to download.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint name under a model family, "
        + "e.g. gpt2, gpt2-medium, gpt2-large, gpt2-xl.",
    )

    parser.add_argument(
        "--download_all_checkpoints",
        action="store_true",
        help="If set to true, downlaod all checkpoitns of a model family.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
    )

    args = parser.parse_args()
    return args


def main():
    """Main"""
    # import pdb; pdb.set_trace()
    args = parse_args()

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

    # Check the validity of the checkpoint
    checkpoints = []
    if args.download_all_checkpoints is True:
        checkpoints = all_checkpoints[args.model_family]
    elif args.checkpoint not in all_checkpoints[args.model_family]:
        print(
            f"Invalid checkpoint from {args.model_family}. Choose from: "
            + f"{all_checkpoints[args.model_family]} or "
            + "set --download_all_checkpoints"
        )
        return
    else:
        checkpoints = [args.checkpoint]

    print(f"Models to download: {checkpoints}")
    for checkpoint in checkpoints:
        print(
            f"Downloading {checkpoint}\t under model family "
            + f"{args.model_family}..."
        )

        # download the model
        # tokenizer = tokenizer_func.from_pretrained(checkpoint)
        # model = model_func.from_pretrained(checkpoint)
        if args.model_family == "Dolly":
            tokenizer = tokenizer_func.from_pretrained(
                checkpoint, padding_side="left"
            )
            model = model_func.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                # torch_dtype=torch.float16,
                # load_in_8bit=True,
            )
        elif args.model_family in [
            "BLIP2-OPT",
            "BLIP2-T5",
            "InstructBLIP",
            "PaliGemma",
            "Idefics2",
        ]:
            tokenizer = tokenizer_func.from_pretrained(checkpoint)
            model = model_func.from_pretrained(
                checkpoint, torch_dtype=torch.float16, device_map="cuda"
            )
        else:
            model = model_func.from_pretrained(checkpoint)
            tokenizer = tokenizer_func.from_pretrained(checkpoint)

        # save the model
        save_dir = os.path.join(args.output_dir, args.model_family, checkpoint)
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        # delete cached files
        # https://huggingface.co/docs/transformers/installation#cache-setup
        cache_dir = "~/.cache/huggingface/hub"

        folders_to_delete = glob.glob(os.path.join(cache_dir, "models*"))
        for folder in folders_to_delete:
            print(f"Removing cached files at {folder}...")
            shutil.rmtree(folder)


if __name__ == "__main__":
    main()
