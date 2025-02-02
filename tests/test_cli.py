import sys
import pytest
import subprocess
from unittest import mock
from unittest.mock import MagicMock, patch
import torch

from mm_poe.cli import main


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    # Mock the inputs provided by questionary
    mock_select.return_value.ask.side_effect = [
        "GIT",  # args.model_family
        "microsoft/git-base-vqav2",  # args.checkpoint
        "FP32",  # args.loading_precision
        "language_modeling",  # args.scoring_method_for_process_of_elimination
        "below_average",  # args.mask_strategy_for_process_of_elimination
        "0",  # args.label
    ]

    mock_path.return_value.ask.side_effect = [
        "./models/",  # args.output_dir
        "./images/image.png",  # args.image_path
    ]

    mock_text.return_value.ask.side_effect = [
        "What is in the image?",  # args.question
        "cat,dog,horse",  # args.choices
    ]

    # Mock the subprocess.call to prevent actual execution
    mock_subprocess_call.return_value = 0

    # Mock the load_model function to return mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    # Mock the device
    with patch("torch.device") as mock_device:
        mock_device.return_value = "cpu"

        # Mock other functions called within main
        with patch("mm_poe.cli.load_data") as mock_load_data, patch(
            "torch.utils.data.DataLoader"
        ) as mock_data_loader_class, patch(
            "mm_poe.cli.inference_language_modeling"
        ) as mock_inference_lm, patch(
            "mm_poe.cli.inference_process_of_elimination"
        ) as mock_inference_poe, patch(
            "mm_poe.cli.compute_mask_process_of_elimination"
        ) as mock_compute_mask, patch(
            "mm_poe.cli.create_multiple_choice_prompt"
        ) as mock_create_mcp:

            # Mock the datasets returned by load_data
            mock_dataset = MagicMock()
            mock_dataset.map.return_value = mock_dataset  # For the map calls
            mock_load_data.return_value = (
                ["hypothesis0", "hypothesis1", "hypothesis2"],  # ending_names
                "premise",  # header_name
                "image_path",  # image_header_name
                mock_dataset,  # raw_dataset
                mock_dataset,  # n_shot_dataset
            )

            # Mock the DataLoader
            mock_data_loader = MagicMock()
            mock_data_loader_class.return_value = mock_data_loader

            # Mock inference functions
            # For scoring_method == 'language_modeling'
            mock_inference_lm.return_value = (
                torch.tensor([[0.1, 0.2, 0.7]]),
                None,
                None,
                torch.tensor([2]),
            )
            # For inference_process_of_elimination
            mock_inference_poe.return_value = (
                torch.tensor([[0.1, 0.2, 0.7]]),
                1.0,
                None,
                torch.tensor([2]),
            )

            # Mock compute_mask_process_of_elimination
            mock_compute_mask.return_value = torch.tensor([[0, 1, 1]])

            # Mock create_multiple_choice_prompt
            def mock_create_mcp_fn(example, **kwargs):
                return example

            mock_create_mcp.side_effect = mock_create_mcp_fn

            # Run the main function
            main()

            # Assertions to check if functions were called as expected
            mock_set_seed.assert_called_once_with(0)
            mock_subprocess_call.assert_called()
            mock_load_model.assert_called()
            mock_load_data.assert_called()
            mock_inference_lm.assert_called()
            mock_inference_poe.assert_called()


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_calibration_lowest(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "BLIP2-OPT",  # args.model_family
        "Salesforce/blip2-opt-2.7b",  # args.checkpoint
        "FP16",  # args.loading_precision
        "calibration",  # args.scoring_method_for_process_of_elimination
        "lowest",  # args.mask_strategy_for_process_of_elimination
        "1",  # args.label
    ]

    mock_path.return_value.ask.side_effect = [
        "./models/",  # args.output_dir
        "./images/image.png",  # args.image_path
    ]

    mock_text.return_value.ask.side_effect = [
        "Describe the image.",  # args.question
        "apple,banana,orange",  # args.choices
    ]

    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cuda:0"

        with patch("mm_poe.cli.load_data") as mock_load_data, patch(
            "torch.utils.data.DataLoader"
        ) as mock_data_loader_class, patch(
            "mm_poe.cli.inference_calibration"
        ) as mock_inference_calibration, patch(
            "mm_poe.cli.inference_process_of_elimination"
        ) as mock_inference_poe, patch(
            "mm_poe.cli.compute_mask_process_of_elimination"
        ) as mock_compute_mask, patch(
            "mm_poe.cli.create_multiple_choice_prompt"
        ) as mock_create_mcp:

            mock_dataset = MagicMock()
            mock_dataset.map.return_value = mock_dataset
            mock_load_data.return_value = (
                ["hypothesis0", "hypothesis1", "hypothesis2"],
                "premise",
                "image_path",
                mock_dataset,
                mock_dataset,
            )

            mock_data_loader = MagicMock()
            mock_data_loader_class.return_value = mock_data_loader

            mock_inference_calibration.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                None,
                None,
                torch.tensor([1]),
            )
            mock_inference_poe.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                1.0,
                None,
                torch.tensor([1]),
            )

            mock_compute_mask.return_value = torch.tensor([[1, 0, 1]])

            def mock_create_mcp_fn(example, **kwargs):
                return example

            mock_create_mcp.side_effect = mock_create_mcp_fn

            main()

            mock_set_seed.assert_called_once_with(0)
            mock_subprocess_call.assert_called()
            mock_load_model.assert_called()
            mock_load_data.assert_called()
            mock_inference_calibration.assert_called()
            mock_inference_poe.assert_called()


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_mcp_lowest(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "BLIP2-OPT",  # args.model_family
        "Salesforce/blip2-opt-2.7b",  # args.checkpoint
        "FP16",  # args.loading_precision
        "multiple_choice_prompt",  # args.scoring_method_for_process_of_elimination
        "lowest",  # args.mask_strategy_for_process_of_elimination
        "1",  # args.label
    ]

    mock_path.return_value.ask.side_effect = [
        "./models/",  # args.output_dir
        "./images/image.png",  # args.image_path
    ]

    mock_text.return_value.ask.side_effect = [
        "Describe the image.",  # args.question
        "apple,banana,orange",  # args.choices
    ]

    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cuda:0"

        with patch("mm_poe.cli.load_data") as mock_load_data, patch(
            "torch.utils.data.DataLoader"
        ) as mock_data_loader_class, patch(
            "mm_poe.cli.inference_language_modeling"
        ) as mock_inference_language_modeling, patch(
            "mm_poe.cli.inference_process_of_elimination"
        ) as mock_inference_poe, patch(
            "mm_poe.cli.compute_mask_process_of_elimination"
        ) as mock_compute_mask, patch(
            "mm_poe.cli.create_multiple_choice_prompt"
        ) as mock_create_mcp:

            mock_dataset = MagicMock()
            mock_dataset.map.return_value = mock_dataset
            mock_load_data.return_value = (
                ["hypothesis0", "hypothesis1", "hypothesis2"],
                "premise",
                "image_path",
                mock_dataset,
                mock_dataset,
            )

            mock_data_loader = MagicMock()
            mock_data_loader_class.return_value = mock_data_loader

            mock_inference_language_modeling.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                None,
                None,
                torch.tensor([1]),
            )
            mock_inference_poe.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                1.0,
                None,
                torch.tensor([1]),
            )

            mock_compute_mask.return_value = torch.tensor([[1, 0, 1]])

            def mock_create_mcp_fn(example, **kwargs):
                return example

            mock_create_mcp.side_effect = mock_create_mcp_fn

            main()

            mock_set_seed.assert_called_once_with(0)
            mock_subprocess_call.assert_called()
            mock_load_model.assert_called()
            mock_load_data.assert_called()
            mock_inference_language_modeling.assert_called()
            mock_inference_poe.assert_called()


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_channel_below_average(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "BLIP2-OPT",  # args.model_family
        "Salesforce/blip2-opt-2.7b",  # args.checkpoint
        "FP16",  # args.loading_precision
        "channel",  # args.scoring_method_for_process_of_elimination
        "below_average",  # args.mask_strategy_for_process_of_elimination
        "1",  # args.label
    ]

    mock_path.return_value.ask.side_effect = [
        "./models/",  # args.output_dir
        "./images/image.png",  # args.image_path
    ]

    mock_text.return_value.ask.side_effect = [
        "Describe the image.",  # args.question
        "apple,banana,orange",  # args.choices
    ]

    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cuda:0"

        with patch("mm_poe.cli.load_data") as mock_load_data, patch(
            "torch.utils.data.DataLoader"
        ) as mock_data_loader_class, patch(
            "mm_poe.cli.inference_language_modeling"
        ) as mock_inference_language_modeling, patch(
            "mm_poe.cli.inference_process_of_elimination"
        ) as mock_inference_poe, patch(
            "mm_poe.cli.compute_mask_process_of_elimination"
        ) as mock_compute_mask, patch(
            "mm_poe.cli.create_multiple_choice_prompt"
        ) as mock_create_mcp:

            mock_dataset = MagicMock()
            mock_dataset.map.return_value = mock_dataset
            mock_load_data.return_value = (
                ["hypothesis0", "hypothesis1", "hypothesis2"],
                "premise",
                "image_path",
                mock_dataset,
                mock_dataset,
            )

            mock_data_loader = MagicMock()
            mock_data_loader_class.return_value = mock_data_loader

            mock_inference_language_modeling.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                None,
                None,
                torch.tensor([1]),
            )
            mock_inference_poe.return_value = (
                torch.tensor([[0.3, 0.4, 0.3]]),
                1.0,
                None,
                torch.tensor([1]),
            )

            mock_compute_mask.return_value = torch.tensor([[1, 0, 1]])

            def mock_create_mcp_fn(example, **kwargs):
                return example

            mock_create_mcp.side_effect = mock_create_mcp_fn

            main()

            mock_set_seed.assert_called_once_with(0)
            mock_subprocess_call.assert_called()
            mock_load_model.assert_called()
            mock_load_data.assert_called()
            mock_inference_language_modeling.assert_called()
            mock_inference_poe.assert_called()


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_mask_strategy_min_k(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "GIT",
        "microsoft/git-base-vqav2",
        "FP32",
        "language_modeling",
        "min_k",
        "0",
    ]
    mock_path.return_value.ask.side_effect = [
        "./models/",
        "./images/image.png",
    ]
    mock_text.return_value.ask.side_effect = [
        "What is in the image?",
        "cat,dog,horse",
    ]
    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cpu"

        # Modify args to include mask_token
        with patch("mm_poe.cli.Namespace") as mock_namespace:
            args = MagicMock()
            args.min_k = 1
            args.process_of_elimination_prompt = (
                "Select the most suitable option "
                + "to answer the question. Ignore [MASK] options."
            )
            mock_namespace.return_value = args

            with patch("mm_poe.cli.load_data") as mock_load_data, patch(
                "torch.utils.data.DataLoader"
            ) as mock_data_loader_class, patch(
                "mm_poe.cli.inference_language_modeling"
            ) as mock_inference_lm, patch(
                "mm_poe.cli.inference_process_of_elimination"
            ) as mock_inference_poe, patch(
                "mm_poe.cli.compute_mask_process_of_elimination"
            ) as mock_compute_mask, patch(
                "mm_poe.cli.create_multiple_choice_prompt"
            ) as mock_create_mcp:

                mock_dataset = MagicMock()
                mock_dataset.map.return_value = mock_dataset
                mock_load_data.return_value = (
                    ["hypothesis0", "hypothesis1", "hypothesis2"],
                    "premise",
                    "image_path",
                    mock_dataset,
                    mock_dataset,
                )

                mock_data_loader = MagicMock()
                mock_data_loader_class.return_value = mock_data_loader

                predictions = torch.tensor([[0.1, 0.2, 0.7]])
                masks = torch.tensor([[0, 1, 1]])
                mock_inference_lm.return_value = (
                    predictions,
                    None,
                    None,
                    torch.tensor([2]),
                )
                mock_inference_poe.return_value = (
                    predictions,
                    1.0,
                    None,
                    torch.tensor([2]),
                )
                mock_compute_mask.return_value = masks

                def mock_create_mcp_fn(example, **kwargs):
                    return example

                mock_create_mcp.side_effect = mock_create_mcp_fn

                main()
                mock_set_seed.assert_called_once_with(0)
                mock_load_model.assert_called()
                mock_load_data.assert_called()
                mock_compute_mask.assert_called_with(
                    predictions, "min_k", min_k=1
                )


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_mask_token(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "GIT",
        "microsoft/git-base-vqav2",
        "FP32",
        "language_modeling",
        "below_average",
        "0",
    ]
    mock_path.return_value.ask.side_effect = [
        "./models/",
        "./images/image.png",
    ]
    mock_text.return_value.ask.side_effect = [
        "What is in the image?",
        "cat,dog,horse",
    ]
    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cpu"

        # Modify args to include mask_token
        with patch("mm_poe.cli.Namespace") as mock_namespace:
            args = MagicMock()
            args.mask_token = "XXX"
            args.process_of_elimination_prompt = (
                "Select the most suitable option "
                + "to answer the question. Ignore [MASK] options."
            )
            mock_namespace.return_value = args

            with patch("mm_poe.cli.load_data") as mock_load_data, patch(
                "torch.utils.data.DataLoader"
            ) as mock_data_loader_class, patch(
                "mm_poe.cli.inference_language_modeling"
            ) as mock_inference_lm, patch(
                "mm_poe.cli.inference_process_of_elimination"
            ) as mock_inference_poe, patch(
                "mm_poe.cli.compute_mask_process_of_elimination"
            ) as mock_compute_mask, patch(
                "mm_poe.cli.create_multiple_choice_prompt"
            ) as mock_create_mcp:

                mock_dataset = MagicMock()
                mock_dataset.map.return_value = mock_dataset
                mock_load_data.return_value = (
                    ["hypothesis0", "hypothesis1", "hypothesis2"],
                    "premise",
                    "image_path",
                    mock_dataset,
                    mock_dataset,
                )

                mock_data_loader = MagicMock()
                mock_data_loader_class.return_value = mock_data_loader

                predictions = torch.tensor([[0.1, 0.2, 0.7]])
                masks = torch.tensor([[0, 1, 1]])
                mock_inference_lm.return_value = (
                    predictions,
                    None,
                    None,
                    torch.tensor([2]),
                )
                mock_inference_poe.return_value = (
                    predictions,
                    1.0,
                    None,
                    torch.tensor([2]),
                )
                mock_compute_mask.return_value = masks

                def mock_create_mcp_fn(example, **kwargs):
                    assert "[MASK]" not in kwargs["multiple_choice_prompt"]
                    return example

                mock_create_mcp.side_effect = mock_create_mcp_fn

                main()
                mock_set_seed.assert_called_once_with(0)
                mock_load_model.assert_called()
                mock_load_data.assert_called()
                mock_compute_mask.assert_called_with(
                    predictions, "below_average"
                )


@patch("mm_poe.cli.set_seed")
@patch("mm_poe.cli.load_model")
@patch("mm_poe.cli.subprocess.call")
@patch("mm_poe.cli.questionary.select")
@patch("mm_poe.cli.questionary.path")
@patch("mm_poe.cli.questionary.text")
def test_main_with_mask_strategy_min_k(
    mock_text,
    mock_path,
    mock_select,
    mock_subprocess_call,
    mock_load_model,
    mock_set_seed,
):
    mock_select.return_value.ask.side_effect = [
        "GIT",
        "microsoft/git-base-vqav2",
        "FP32",
        "language_modeling",
        "min_k",
        "0",
    ]
    mock_path.return_value.ask.side_effect = [
        "./models/",
        "./images/image.png",
    ]
    mock_text.return_value.ask.side_effect = [
        "What is in the image?",
        "cat,dog,horse",
    ]
    mock_subprocess_call.return_value = 0
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_model.return_value = (mock_model, mock_tokenizer)

    with patch("torch.device") as mock_device:
        mock_device.return_value = "cpu"

        # Modify args to include mask_token
        with patch("mm_poe.cli.Namespace") as mock_namespace:
            args = MagicMock()
            args.min_k = 10
            args.process_of_elimination_prompt = (
                "Select the most suitable "
                + "option to answer the question. Ignore [MASK] options."
            )
            mock_namespace.return_value = args

            with patch("mm_poe.cli.load_data") as mock_load_data, patch(
                "torch.utils.data.DataLoader"
            ) as mock_data_loader_class, patch(
                "mm_poe.cli.inference_language_modeling"
            ) as mock_inference_lm, patch(
                "mm_poe.cli.inference_process_of_elimination"
            ) as mock_inference_poe, patch(
                "mm_poe.cli.compute_mask_process_of_elimination"
            ) as mock_compute_mask, patch(
                "mm_poe.cli.create_multiple_choice_prompt"
            ) as mock_create_mcp:

                mock_dataset = MagicMock()
                mock_dataset.map.return_value = mock_dataset
                mock_load_data.return_value = (
                    ["hypothesis0", "hypothesis1", "hypothesis2"],
                    "premise",
                    "image_path",
                    mock_dataset,
                    mock_dataset,
                )

                mock_data_loader = MagicMock()
                mock_data_loader_class.return_value = mock_data_loader

                predictions = torch.tensor([[0.1, 0.2, 0.7]])
                masks = torch.tensor([[0, 1, 1]])
                mock_inference_lm.return_value = (
                    predictions,
                    None,
                    None,
                    torch.tensor([2]),
                )
                mock_inference_poe.return_value = (
                    predictions,
                    1.0,
                    None,
                    torch.tensor([2]),
                )
                mock_compute_mask.return_value = masks

                def mock_create_mcp_fn(example, **kwargs):
                    return example

                mock_create_mcp.side_effect = mock_create_mcp_fn

                main()
                mock_set_seed.assert_called_once_with(0)
                mock_load_model.assert_called()
                mock_load_data.assert_called()
                mock_compute_mask.assert_called_with(
                    predictions, "min_k", min_k=2
                )
