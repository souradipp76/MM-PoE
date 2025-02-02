import os
import sys
import random
import numpy as np
import torch
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open

# Import the functions from utils.py
from mm_poe.methods.utils.utils import (
    set_seed,
    parse_args,
    load_data,
    load_model,
    write_to_csv,
)


# Test for set_seed function
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_set_seed(seed):
    with mock.patch("torch.cuda.manual_seed_all") as mock_cuda_seed_all:
        # Call the function with the seed
        set_seed(seed)

        # Check if os environment variable is set correctly
        assert os.environ["PYTHONHASHSEED"] == str(seed)

        # Check if random seeds are set correctly
        random_value = random.randint(0, 100)
        np_value = np.random.randint(0, 100)
        torch_value = torch.randint(0, 100, (1,)).item()

        set_seed(seed)  # Reset seeds

        assert random.randint(0, 100) == random_value
        assert np.random.randint(0, 100) == np_value
        assert torch.randint(0, 100, (1,)).item() == torch_value

        # Check if CUDA seeds were set correctly (mocked)
        mock_cuda_seed_all.assert_called_with(seed)


# Tests for parse_args function
def test_parse_args_with_required_arguments():
    test_args = [
        "script_name",
        "--model_family",
        "GPT2",
        "--checkpoint",
        "gpt2-medium",
        "--datasets",
        "copa",
    ]
    with mock.patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.model_family == "GPT2"
        assert args.checkpoint == "gpt2-medium"
        assert args.datasets == "copa"


def test_parse_args_with_all_arguments():
    test_args = [
        "script_name",
        "--model_family",
        "GPT2",
        "--checkpoint",
        "gpt2-medium",
        "--datasets",
        "copa winogrande",
        "--seed",
        "42",
        "--amateur_checkpoint",
        "gpt2-small",
        "--expert_method",
        "language_modeling",
        "--amateur_method",
        "calibration",
        "--weighting_parameter",
        "-0.5",
        "--weighting_parameters",
        "0.1,0.2",
        "--num_random_search",
        "5",
        "--loading_precision",
        "FP16",
        "--sample",
        "100",
        "--batch_size",
        "16",
        "--n_shot",
        "5",
        "--multiple_choice_prompt",
        "Choose the best option:",
        "--calibration_prompt",
        "This is a calibration prompt.",
        "--do_channel",
        "--process_of_elimination_prompt",
        "Eliminate incorrect options:",
        "--scoring_method_for_process_of_elimination",
        "calibration",
        "--prompting_method_for_process_of_elimination",
        "multiple_choice_prompt",
        "--mask_strategy_for_process_of_elimination",
        "min_k",
        "--do_synonym",
        "--number_of_synonyms",
        "3",
        "--generate_synonyms_prompt",
        "Generate synonyms for option: option",
        "--push_data_to_hub",
        "--min_k",
        "2",
        "--mask_token",
        "[MASK]",
    ]
    with mock.patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.seed == 42
        assert args.amateur_checkpoint == "gpt2-small"
        assert args.expert_method == "language_modeling"
        assert args.amateur_method == "calibration"
        assert args.weighting_parameter == -0.5
        assert args.weighting_parameters == "0.1,0.2"
        assert args.num_random_search == 5
        assert args.loading_precision == "FP16"
        assert args.sample == 100
        assert args.batch_size == 16
        assert args.n_shot == 5
        assert args.multiple_choice_prompt == "Choose the best option:"
        assert args.calibration_prompt == "This is a calibration prompt."
        assert args.do_channel is True
        assert (
            args.process_of_elimination_prompt
            == "Eliminate incorrect options:"
        )
        assert args.scoring_method_for_process_of_elimination == "calibration"
        assert (
            args.prompting_method_for_process_of_elimination
            == "multiple_choice_prompt"
        )
        assert args.mask_strategy_for_process_of_elimination == "min_k"
        assert args.do_synonym is True
        assert args.number_of_synonyms == 3
        assert (
            args.generate_synonyms_prompt
            == "Generate synonyms for option: option"
        )
        assert args.push_data_to_hub is True
        assert args.min_k == 2
        assert args.mask_token == "[MASK]"


def test_parse_args_missing_required_arguments():
    test_args = [
        "script_name",
        "--model_family",
        "GPT2",
        # "--checkpoint" is missing
        "--datasets",
        "copa",
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            parse_args()


# Tests for load_data function
@pytest.mark.parametrize(
    "dataset_name,loader_name,ending_names,header_name",
    [
        ("copa", "copa_loader", ["hypothesis0", "hypothesis1"], "premise"),
        (
            "cqa",
            "cqa_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "obqa",
            "obqa_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2", "hypothesis3"],
            "premise",
        ),
        ("piqa", "piqa_loader", ["hypothesis0", "hypothesis1"], "premise"),
        (
            "qasc",
            "qasc_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
                "hypothesis5",
                "hypothesis6",
                "hypothesis7",
            ],
            "premise",
        ),
        (
            "siqa",
            "siqa_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2"],
            "premise",
        ),
        (
            "winogrande",
            "winogrande_loader",
            ["hypothesis0", "hypothesis1"],
            "premise",
        ),
        (
            "anli",
            "anli_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2"],
            "premise",
        ),
        (
            "disambiguation_qa",
            "date_understanding_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2"],
            "premise",
        ),
        (
            "conceptual_combinations",
            "date_understanding_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2", "hypothesis3"],
            "premise",
        ),
        (
            "date_understanding",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
                "hypothesis5",
            ],
            "premise",
        ),
        (
            "emoji_movie",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "ruin_names",
            "date_understanding_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2", "hypothesis3"],
            "premise",
        ),
        (
            "penguins_in_a_table",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "strange_stories",
            "date_understanding_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2", "hypothesis3"],
            "premise",
        ),
        (
            "reasoning_about_colored_objects",
            "date_understanding_loader",
            [f"hypothesis{i}" for i in range(18)],
            "premise",
        ),
        (
            "symbol_interpretation",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "tracking_shuffled_objects",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "logical_deduction_three_objects",
            "date_understanding_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2"],
            "premise",
        ),
        (
            "logical_deduction_five_objects",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
            ],
            "premise",
        ),
        (
            "logical_deduction_seven_objects",
            "date_understanding_loader",
            [
                "hypothesis0",
                "hypothesis1",
                "hypothesis2",
                "hypothesis3",
                "hypothesis4",
                "hypothesis5",
                "hypothesis6",
            ],
            "premise",
        ),
        (
            "anli_r1",
            "anli_loader",
            ["hypothesis0", "hypothesis1", "hypothesis2"],
            "premise",
        ),
        (
            "vqa",
            "vqa_loader",
            [f"hypothesis{i}" for i in range(18)],
            "premise",
        ),
        (
            "scienceqa",
            "scienceqa_loader",
            [f"hypothesis{i}" for i in range(4)],
            "premise",
        ),
        (
            "ai2d",
            "ai2d_loader",
            [f"hypothesis{i}" for i in range(4)],
            "premise",
        ),
        (
            "single_inference",
            "single_inference_loader",
            [f"hypothesis{i}" for i in range(4)],
            "premise",
        ),
    ],
)
def test_load_data_datasets(
    dataset_name, loader_name, ending_names, header_name
):
    # Create a mock args object
    class Args:
        dataset = dataset_name
        image_path = None
        sample = None
        n_shot = 0
        num_options = len(ending_names)

    args = Args()

    # Mock the data loader function
    loader_path = f"mm_poe.methods.utils.utils.{loader_name}"
    with mock.patch(loader_path) as mock_loader:
        # Mock return value
        mock_value = {
            "premise": "Test premise",
            "uncond_premise": "Test premise",
            "image_path": "dummy_path",
            "label": 0,
        }
        for i, ending_name in enumerate(ending_names):
            mock_value[ending_name] = f"answer {i}"
        mock_loader.return_value = [mock_value]
        # Mock os.path.join to prevent file system access
        with mock.patch("os.path.join", return_value="dummy_path"):
            if dataset_name in [
                "vqa",
                "scienceqa",
                "ai2d",
                "single_inference",
            ]:
                ending, header, image_header, dev_dataset, train_dataset = (
                    load_data(args)
                )
                assert image_header == "image_path"
            else:
                ending, header, dev_dataset, train_dataset = load_data(args)
            assert ending == ending_names
            assert header == header_name
            assert len(dev_dataset) == 1
            assert len(train_dataset) == 1


def test_load_data_invalid_dataset():
    class Args:
        dataset = "unknown_dataset"

    args = Args()

    with mock.patch("builtins.print") as mock_print:
        result = load_data(args)
        assert result is None
        mock_print.assert_called_with(
            f"{args.dataset}: downloader not implemented."
        )


# Tests for load_model function
@pytest.mark.parametrize(
    "model_family,model_func_name,tokenizer_func_name",
    [
        ("GPT2", "AutoModelForCausalLM", "AutoTokenizer"),
        ("Pythia", "AutoModelForCausalLM", "AutoTokenizer"),
        ("OPT-IML", "AutoModelForCausalLM", "AutoTokenizer"),
        ("Dolly", "AutoModelForCausalLM", "AutoTokenizer"),
        ("T5", "AutoModelForSeq2SeqLM", "AutoTokenizer"),
        ("FLAN-T5", "AutoModelForSeq2SeqLM", "AutoTokenizer"),
        ("BLIP2-OPT", "AutoModelForVision2Seq", "AutoProcessor"),
        ("BLIP2-T5", "AutoModelForVision2Seq", "AutoProcessor"),
        ("InstructBLIP", "AutoModelForVision2Seq", "AutoProcessor"),
        ("GIT", "AutoModelForVision2Seq", "AutoProcessor"),
        ("PaliGemma", "AutoModelForVision2Seq", "AutoProcessor"),
        ("Idefics2", "AutoModelForVision2Seq", "AutoProcessor"),
    ],
)
def test_load_model_families(
    model_family, model_func_name, tokenizer_func_name
):
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:

        model_family = ""
        loading_precision = "FP32"

        def __init__(self, model_family):
            self.model_family = model_family

    args = Args(model_family)

    # Mock the tokenizer and model loading functions
    with mock.patch(
        f"mm_poe.methods.utils.utils.{tokenizer_func_name}"
    ) as mock_tokenizer_class:
        with mock.patch(
            f"mm_poe.methods.utils.utils.{model_func_name}"
        ) as mock_model_class:
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            # Set the return value of get_memory_footprint to a numeric value
            mock_model.get_memory_footprint.return_value = (
                2 * 1024**3
            )  # 2 GB in bytes

            model, tokenizer = load_model(device, model_path, args)

            # Check that the correct tokenizer and model are loaded
            if model_family == "Dolly":
                mock_tokenizer_class.from_pretrained.assert_called_with(
                    model_path, padding_side="left"
                )
            elif model_family == "Idefics2":
                mock_tokenizer_class.from_pretrained.assert_called_with(
                    model_path, do_image_splitting=False
                )
            else:
                mock_tokenizer_class.from_pretrained.assert_called_with(
                    model_path
                )


def test_load_model_invalid_family():
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:
        model_family = "UnknownFamily"
        loading_precision = "FP32"

    args = Args()

    with mock.patch("builtins.print") as mock_print:
        result = load_model(device, model_path, args)
        assert result is None
        mock_print.assert_called_with(
            f"{args.model_family}: downloader not implemented."
        )


def test_load_model_loading_precision_int8():
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:
        model_family = "GPT2"
        loading_precision = "INT8"

    args = Args()

    # Mock the tokenizer and model loading functions
    with mock.patch(
        "mm_poe.methods.utils.utils.AutoTokenizer"
    ) as mock_tokenizer_class:
        with mock.patch(
            "mm_poe.methods.utils.utils.AutoModelForCausalLM"
        ) as mock_model_class:
            with mock.patch(
                "mm_poe.methods.utils.utils.BitsAndBytesConfig"
            ) as mock_bnb_config_class:
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_bnb_config = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = (
                    mock_tokenizer
                )
                mock_model_class.from_pretrained.return_value = mock_model
                mock_bnb_config_class.return_value = mock_bnb_config

                # Set the return value of get_memory_footprint to a numeric value
                mock_model.get_memory_footprint.return_value = (
                    2 * 1024**3
                )  # 2 GB in bytes

                model, tokenizer = load_model(device, model_path, args)

                # Check that BitsAndBytesConfig is called correctly
                mock_bnb_config_class.assert_called_with(
                    load_in_8bit=True, llm_int8_threshold=200.0
                )
                # Check that model is loaded with quantization config
                mock_model_class.from_pretrained.assert_called_with(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=device,
                    quantization_config=mock_bnb_config,
                )


def test_load_model_loading_precision_int4():
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:
        model_family = "GPT2"
        loading_precision = "INT4"

    args = Args()

    # Mock the tokenizer and model loading functions
    with mock.patch(
        "mm_poe.methods.utils.utils.AutoTokenizer"
    ) as mock_tokenizer_class:
        with mock.patch(
            "mm_poe.methods.utils.utils.AutoModelForCausalLM"
        ) as mock_model_class:
            with mock.patch(
                "mm_poe.methods.utils.utils.BitsAndBytesConfig"
            ) as mock_bnb_config_class:
                mock_tokenizer = MagicMock()
                mock_model = MagicMock()
                mock_bnb_config = MagicMock()
                mock_tokenizer_class.from_pretrained.return_value = (
                    mock_tokenizer
                )
                mock_model_class.from_pretrained.return_value = mock_model
                mock_bnb_config_class.return_value = mock_bnb_config

                # Set the return value of get_memory_footprint to a numeric value
                mock_model.get_memory_footprint.return_value = (
                    2 * 1024**3
                )  # 2 GB in bytes

                model, tokenizer = load_model(device, model_path, args)

                # Check that BitsAndBytesConfig is called correctly
                mock_bnb_config_class.assert_called_with(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                # Check that model is loaded with quantization config
                mock_model_class.from_pretrained.assert_called_with(
                    model_path,
                    device_map=device,
                    quantization_config=mock_bnb_config,
                )


def test_load_model_loading_precision_fp16():
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:
        model_family = "GPT2"
        loading_precision = "FP16"

    args = Args()

    # Mock the tokenizer and model loading functions
    with mock.patch(
        "mm_poe.methods.utils.utils.AutoTokenizer"
    ) as mock_tokenizer_class:
        with mock.patch(
            "mm_poe.methods.utils.utils.AutoModelForCausalLM"
        ) as mock_model_class:
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_bnb_config = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            # Set the return value of get_memory_footprint to a numeric value
            mock_model.get_memory_footprint.return_value = (
                2 * 1024**3
            )  # 2 GB in bytes

            model, tokenizer = load_model(device, model_path, args)

            # Check that model is loaded with quantization config
            mock_model_class.from_pretrained.assert_called_with(
                model_path, torch_dtype=torch.float16, device_map=device
            )


def test_load_model_loading_precision_bf16():
    device = "cpu"
    model_path = "some-model-path"

    # Create a mock args object
    class Args:
        model_family = "GPT2"
        loading_precision = "BF16"

    args = Args()

    # Mock the tokenizer and model loading functions
    with mock.patch(
        "mm_poe.methods.utils.utils.AutoTokenizer"
    ) as mock_tokenizer_class:
        with mock.patch(
            "mm_poe.methods.utils.utils.AutoModelForCausalLM"
        ) as mock_model_class:
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_bnb_config = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            # Set the return value of get_memory_footprint to a numeric value
            mock_model.get_memory_footprint.return_value = (
                2 * 1024**3
            )  # 2 GB in bytes

            model, tokenizer = load_model(device, model_path, args)

            # Check that model is loaded with quantization config
            mock_model_class.from_pretrained.assert_called_with(
                model_path, torch_dtype=torch.bfloat16, device_map=device
            )


# Tests for write_to_csv function
def test_write_to_csv_process_of_elimination(tmp_path):
    save_path = tmp_path / "results.csv"

    class Args:
        method = "process_of_elimination"
        model_family = "GPT2"
        checkpoint = "gpt2-medium"
        loading_precision = "FP32"
        dataset = "copa"
        batch_size = 32
        scoring_method_for_process_of_elimination = "language_modeling"
        prompting_method_for_process_of_elimination = "multiple_choice_prompt"
        mask_strategy_for_process_of_elimination = "lowest"
        mask_token = None
        seed = 42
        n_shot = 0
        sample = None
        mask_accuracy = 0.8

    args = Args()
    total_accuracy = 0.85

    write_to_csv(str(save_path), args, total_accuracy)

    with open(save_path, "r") as f:
        content = f.read()
        assert "process_of_elimination" in content
        assert f"{args.mask_accuracy:.4f}" in content
        assert f"{total_accuracy:.4f}" in content


def test_write_to_csv_contrastive_decoding(tmp_path):
    save_path = tmp_path / "results.csv"

    class Args:
        method = "contrastive_decoding"
        model_family = "GPT2"
        checkpoint = "gpt2-medium"
        amateur_checkpoint = "gpt2-small"
        loading_precision = "FP32"
        dataset = "copa"
        batch_size = 32
        expert_method = "language_modeling"
        amateur_method = "calibration"
        weighting_parameter = -1.0
        seed = 42
        n_shot = 0
        sample = None
        expert_accuracy = 0.9
        amateur_accuracy = 0.7

    args = Args()
    total_accuracy = 0.85

    write_to_csv(str(save_path), args, total_accuracy)

    with open(save_path, "r") as f:
        content = f.read()
        assert "contrastive_decoding" in content
        assert f"{args.expert_accuracy:.4f}" in content
        assert f"{args.amateur_accuracy:.4f}" in content
        assert f"{total_accuracy:.4f}" in content


def test_write_to_csv_generate_synonyms(tmp_path):
    save_path = tmp_path / "results.csv"

    class Args:
        method = "generate_synonyms"
        model_family = "GPT2"
        checkpoint = "gpt2-medium"
        loading_precision = "FP32"
        dataset = "copa"
        batch_size = 32
        number_of_synonyms = 5
        seed = 42
        n_shot = 0
        sample = None

    args = Args()
    total_accuracy = 0.88

    write_to_csv(str(save_path), args, total_accuracy)

    with open(save_path, "r") as f:
        content = f.read()
        assert "generate_synonyms" in content
        assert f"{total_accuracy:.4f}" in content


def test_write_to_csv_default_method(tmp_path):
    save_path = tmp_path / "results.csv"

    class Args:
        method = "default_method"
        model_family = "GPT2"
        checkpoint = "gpt2-medium"
        loading_precision = "FP32"
        dataset = "copa"
        batch_size = 32
        seed = 42
        n_shot = 0
        sample = None

    args = Args()
    total_accuracy = 0.9

    write_to_csv(str(save_path), args, total_accuracy)

    with open(save_path, "r") as f:
        content = f.read()
        assert "default_method" in content
        assert f"{total_accuracy:.4f}" in content


# Additional tests for branches and edge cases
def test_parse_args_invalid_choice():
    test_args = [
        "script_name",
        "--model_family",
        "GPT2",
        "--checkpoint",
        "gpt2-medium",
        "--datasets",
        "copa",
        "--loading_precision",
        "INVALID_PRECISION",
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            parse_args()


def test_write_to_csv_no_method(tmp_path):
    save_path = tmp_path / "results.csv"

    class Args:
        method = None
        model_family = "GPT2"
        checkpoint = "gpt2-medium"
        loading_precision = "FP32"
        dataset = "copa"
        batch_size = 32
        seed = 42
        n_shot = 0
        sample = 100

    args = Args()
    total_accuracy = 0.9

    # with pytest.raises(AttributeError):
    write_to_csv(str(save_path), args, total_accuracy)
    assert os.path.isfile(save_path) == True
