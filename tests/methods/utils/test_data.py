# test_data.py

import os
import sys
import torch
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch, Mock
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import xml.etree.ElementTree as ET
from PIL import Image
import random

# Import all functions from data.py
# Adjust the import path based on your project structure
# Assuming data.py is in the same directory as test_data.py
from mm_poe.methods.utils.data import (
    upload_to_huggingface_hub,
    preprocess_function_seq2seq,
    preprocess_function_causal,
    preprocess_function_seq2seq_vqa,
    preprocess_function_causal_vqa,
    preprocess_function_seq2seq_channel,
    preprocess_function_causal_channel,
    preprocess_function_seq2seq_vqa_channel,
    preprocess_function_causal_vqa_channel,
    create_multiple_choice_prompt,
    create_synonym_dataset,
    copa_loader,
    cqa_loader,
    obqa_loader,
    piqa_loader,
    qasc_loader,
    siqa_loader,
    winogrande_loader,
    date_understanding_loader,
    anli_loader,
    generate_n_shot_demonstrations,
    create_n_shot_splits,
    generate_n_shot_poe_demonstrations,
    vqa_loader,
    scienceqa_loader,
    ai2d_loader,
    single_inference_loader,
)


# Mock class for argparse.Namespace
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def sample_args():
    return Args(
        dataset="test_dataset",
        seed=42,
        n_shot=5,
        sample=10,
        checkpoint="checkpoints/test_checkpoint",
        batch_size=32,
        method="test_method",
        ending_names=["choice0", "choice1", "choice2", "choice3"],
        header_name="question",
        tokenizer=Mock(),
        processor=Mock(),
        image_header_name="image_path",
        multiple_choice_prompt="Please select the correct answer:",
        scoring_method="other_method",
        num_of_options=3,
        mask_token=None,
        number_of_synonyms=2,
        calibration_prompt=" the answer is:",
        num_options=4,
        mask=[1, 0, 1],
        # sample=None,
        n_shot_demonstrations="",
        image_processor=Mock(),
        synonyms_dict={
            "Paris": ["Paris1", "Paris2"],
            "London": ["London1", "London2"],
        },
        question="What is the capital of France?",
        choices=["Paris", "London", "Berlin"],
        label=0,
    )


def test_upload_to_huggingface_hub(sample_args):
    dataset = MagicMock()
    args = sample_args
    suffix = f"{args.dataset}_{args.seed}_{args.n_shot}_{args.sample}_{args.checkpoint.split('/')[-1]}_{args.batch_size}"
    temp_data_path = os.path.join(f"../temp_data/{args.method}", suffix)

    with patch("os.system") as mock_system:
        upload_to_huggingface_hub(dataset, args)
        dataset.save_to_disk.assert_called_once_with(temp_data_path)


def test_preprocess_function_seq2seq(sample_args):
    examples = {
        "question": ["What is the capital of France?", "What is 2+2?"],
        "choice0": ["Paris", "3"],
        "choice1": ["London", "4"],
        "choice2": ["Berlin", "5"],
        "choice3": ["Madrid", "6"],
    }
    tokenizer = MagicMock()
    # Adjust the tokenizer mock to return the correct number of tokens
    tokenizer.return_value = {
        "input_ids": [[i] for i in range(8)],  # 2 questions * 4 choices = 8
        "attention_mask": [[1] for _ in range(8)],
    }
    kwargs = {
        "ending_names": ["choice0", "choice1", "choice2", "choice3"],
        "header_name": "question",
        "tokenizer": tokenizer,
    }
    output = preprocess_function_seq2seq(examples, **kwargs)
    assert "header_input_ids" in output
    assert "header_attention_mask" in output
    assert "ending_input_ids" in output
    assert "ending_attention_mask" in output
    num_choice = 4
    for key in output:
        assert len(output[key]) == len(examples["question"])
        for sublist in output[key]:
            assert len(sublist) == num_choice


def test_preprocess_function_causal(sample_args):
    examples = {
        "question": ["What is the capital of France?"],
        "choice0": ["Paris"],
        "choice1": ["London"],
    }
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    # Fix the lambda function to define 's'
    tokenizer.side_effect = lambda x, truncation: (
        {
            "input_ids": [list(range(len(s))) for s in x],
            "attention_mask": [[1] * len(s) for s in x],
        }
        if isinstance(x, list)
        else {}
    )
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "tokenizer": tokenizer,
    }
    output = preprocess_function_causal(examples, **kwargs)
    assert "input_ids" in output
    assert "labels" in output
    assert "ending_attention_mask" in output


def test_preprocess_function_seq2seq_vqa(sample_args):
    examples = {
        "question": ["What is shown in the image?"],
        "choice0": ["Cat"],
        "choice1": ["Dog"],
        "image_path": ["path/to/image1.jpg"],
    }
    processor = MagicMock()
    # Adjust the tokenizer and image processor mocks
    processor.tokenizer.return_value = {
        "input_ids": [[i] for i in range(2)],
        "attention_mask": [[1] for _ in range(2)],
    }
    data_obj = MagicMock()
    data_obj.data = {
        "pixel_values": torch.tensor(
            [[[1, 2], [3, 4]]] * 2
        )  # Repeat to match the number of choices
    }
    processor.image_processor.return_value = data_obj
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "image_header_name": "image_path",
        "processor": processor,
        "image_token": "",
    }
    with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
        output = preprocess_function_seq2seq_vqa(examples, **kwargs)
        assert "header_input_ids" in output
        assert "ending_input_ids" in output
        assert "images" in output
        assert len(output["images"]) == len(examples["question"])
        for img_list in output["images"]:
            assert len(img_list) == len(kwargs["ending_names"])


def test_create_multiple_choice_prompt(sample_args):
    example = {
        "premise": "What is the capital of France?",
        "uncond_premise": "The answer is:",
        "hypothesis0": "Paris",
        "hypothesis1": "London",
        "hypothesis2": "Berlin",
        "mask": [1, 0, 1],
    }
    kwargs = {
        "multiple_choice_prompt": "Please choose the correct answer:",
        "scoring_method": "other_method",
        "num_of_options": 3,
        "mask_token": None,
    }
    output = create_multiple_choice_prompt(example, **kwargs)
    expected_premise = "Please choose the correct answer:\n Question: What is the capital of France?\nA. Paris\nB. [MASK]\nC. Berlin\nAnswer:"
    assert output["premise"] == expected_premise
    kwargs["scoring_method"] = "multiple_choice_prompt"
    example["premise"] = (
        " Question: What is the capital of France?\nA. Paris\nB. London\nC. Berlin\nAnswer:"
    )
    output = create_multiple_choice_prompt(example, **kwargs)
    assert output["premise"] == expected_premise


def test_create_synonym_dataset(sample_args):
    examples = {
        "hypothesis0": ["Paris", "London"],
        "hypothesis1": ["Berlin", "Madrid"],
    }
    kwargs = {
        "args": sample_args,
        "synonyms_dict": {
            "Paris": ["Paris1", "Paris2"],
            "London": ["London1", "London2"],
            "Berlin": ["Berlin1", "Berlin2"],
            "Madrid": ["Madrid1", "Madrid2"],
        },
    }
    output = create_synonym_dataset(examples, **kwargs)
    for hypothesis in ["hypothesis0", "hypothesis1"]:
        for i in range(sample_args.number_of_synonyms):
            key = f"{hypothesis}_synonyms_{i}"
            assert key in output
            assert len(output[key]) == len(examples[hypothesis])


def test_copa_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = None
    xml_content = """<root>
    <item most-plausible-alternative="1" asks-for="effect">
        <p>It started to rain.</p>
        <a1>I opened my umbrella.</a1>
        <a2>I wore sunglasses.</a2>
    </item>
    </root>"""
    with patch("xml.etree.ElementTree.parse") as mock_parse:
        mock_tree = ET.ElementTree(ET.fromstring(xml_content))
        mock_parse.return_value = mock_tree
        examples = copa_loader("dummy_path.xml", args)
        assert len(examples) == 1
        assert examples[0]["label"] == 0
        assert examples[0]["premise"] == " It started to rain so"
        assert examples[0]["hypothesis0"] == " i opened my umbrella."
        assert examples[0]["hypothesis1"] == " i wore sunglasses."


def test_copa_loader_assert(sample_args):
    args = sample_args
    xml_content = """<root>
    <item most-plausible-alternative="1" asks-for="unknown">
        <p>It started to rain.</p>
        <a1>I opened my umbrella.</a1>
        <a2>I wore sunglasses.</a2>
    </item>
    </root>"""
    with pytest.raises(AssertionError):
        with patch("xml.etree.ElementTree.parse") as mock_parse:
            mock_tree = ET.ElementTree(ET.fromstring(xml_content))
            mock_parse.return_value = mock_tree
            examples = copa_loader("dummy_path.xml", args)


def test_cqa_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    # Adjust the stem to end with a period to match the processing in cqa_loader
    json_line = json.dumps(
        {
            "answerKey": "A",
            "question": {
                "stem": "What is the capital of France.",
                "choices": [
                    {"text": "Paris"},
                    {"text": "London"},
                    {"text": "Berlin"},
                    {"text": "Madrid"},
                    {"text": "Rome"},
                ],
            },
        }
    )
    with patch("builtins.open", mock.mock_open(read_data=json_line)):
        examples = cqa_loader("dummy_path.jsonl", args)
        assert len(examples) == 1
        assert examples[0]["label"] == 0
        assert (
            "Answer the following question: Question:  What is the capital of France?"
            in examples[0]["premise"]
        )


def test_obqa_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    # Adjust the stem to end with a period to match the processing in cqa_loader
    json_line = json.dumps(
        {
            "answerKey": "A",
            "question": {
                "stem": "What is the capital of France.",
                "choices": [
                    {"text": "Paris", "label": 0},
                    {"text": "London", "label": 0},
                    {"text": "Berlin", "label": 1},
                    {"text": "Madrid", "label": 0},
                    {"text": "Rome", "label": 0},
                ],
            },
        }
    )
    with patch("builtins.open", mock.mock_open(read_data=json_line)):
        examples = obqa_loader("dummy_path.jsonl", args)
        assert len(examples) == 1
        assert examples[0]["label"] == 0
        assert (
            "Answer the following question: Question: What is the capital of France"
            in examples[0]["premise"]
        )


def test_generate_n_shot_demonstrations(sample_args):
    n_shot_dataset = [
        {
            "premise": "Question 1",
            "label": torch.tensor(0),
            "hypothesis0": "A1",
            "hypothesis1": "B1",
        },
        {
            "premise": "Question 2",
            "label": torch.tensor(1),
            "hypothesis0": "A2",
            "hypothesis1": "B2",
        },
    ]
    output = generate_n_shot_demonstrations(n_shot_dataset)
    expected_output = "Question 1A1\n\nQuestion 2B2\n\n"
    assert output == expected_output


def test_create_n_shot_splits(sample_args):
    args = sample_args
    args.n_shot = 1
    raw_dataset = MagicMock()
    n_shot_dataset = MagicMock()
    n_shot_dataset.shuffle.return_value.select.return_value = n_shot_dataset
    raw_dataset.shuffle.return_value.select.return_value = raw_dataset
    raw_dataset.map.return_value = raw_dataset
    # Adjust the patch path to match the module structure
    with patch(
        "mm_poe.methods.utils.data.generate_n_shot_demonstrations",
        return_value="Demo",
    ) as mock_generate:
        output_dataset, output_n_shot_dataset, n_shot_demonstrations = (
            create_n_shot_splits(raw_dataset, n_shot_dataset, args)
        )
        assert n_shot_demonstrations == "Demo"
        raw_dataset.map.assert_called_once()


def test_single_inference_loader(sample_args):
    args = sample_args
    path = "path/to/image.jpg"
    examples = single_inference_loader(path, args)
    assert len(examples) == 1
    assert examples[0]["image_path"] == path
    assert examples[0]["premise"].startswith(args.multiple_choice_prompt)


def test_anli_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = None
    json_line = json.dumps(
        {
            "context": "A man is playing a piano.",
            "hypothesis": "The man is playing a musical instrument.",
            "label": "e",
        }
    )
    with patch("builtins.open", mock.mock_open(read_data=json_line)):
        examples = anli_loader(["dummy_path.jsonl"], args)
        assert len(examples) == 1
        assert examples[0]["label"] == 0
        assert (
            "A man is playing a piano. The man is playing a musical instrument."
            in examples[0]["premise"]
        )


def test_generate_n_shot_poe_demonstrations(sample_args):
    n_shot_dataset = [
        {
            "premise": "Question 1",
            "label": torch.tensor(0),
            "hypothesis0": "A1",
            "hypothesis1": "B1",
        },
        {
            "premise": "Question 2",
            "label": torch.tensor(1),
            "hypothesis0": "A2",
            "hypothesis1": "B2",
        },
    ]
    num_of_options = 2
    output, poe_output = generate_n_shot_poe_demonstrations(
        n_shot_dataset, num_of_options
    )
    assert isinstance(output, str)
    assert isinstance(poe_output, str)


def test_preprocess_function_seq2seq_channel(sample_args):
    examples = {
        "question": ["What is 2+2?"],
        "choice0": ["3"],
        "choice1": ["4"],
    }
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "tokenizer": tokenizer,
    }
    output = preprocess_function_seq2seq_channel(examples, **kwargs)
    assert "header_input_ids" in output
    assert "ending_input_ids" in output


def test_preprocess_function_causal_channel(sample_args):
    examples = {
        "question": ["What is 2+2?"],
        "choice0": ["3"],
        "choice1": ["4"],
    }
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    # Adjust the tokenizer to return lists of lists
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
    }
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "tokenizer": tokenizer,
    }
    output = preprocess_function_causal_channel(examples, **kwargs)
    assert "input_ids" in output
    assert "labels" in output


def test_vqa_loader(sample_args):
    args = sample_args
    args.num_options = 2
    args.split = "val"
    ann_content = {
        "annotations": [{"multiple_choice_answer": "cat", "image_id": 123}]
    }
    ques_content = {
        "questions": [
            {
                "question": "What animal is this?",
                "multiple_choices": ["cat", "dog"],
            }
        ]
    }
    with patch("json.load", side_effect=[ann_content, ques_content]):
        with patch("os.path.join", return_value="path/to/image.jpg"):
            with patch("builtins.open", mock.mock_open()) as mock_file:
                # Mock the open calls for annotation and question files
                mock_file.side_effect = [
                    mock.mock_open(
                        read_data=json.dumps(ann_content)
                    ).return_value,
                    mock.mock_open(
                        read_data=json.dumps(ques_content)
                    ).return_value,
                ]
                examples = vqa_loader("dummy_path", args)
                assert len(examples) == 1
                assert examples[0]["label"] == 0
                assert examples[0]["image_path"] == "path/to/image.jpg"


def test_scienceqa_loader(sample_args):
    args = sample_args
    args.num_options = 4
    args.split = "val"
    ann_content = {
        "1": {
            "question": "What is H2O?",
            "choices": ["Water", "Oxygen", "Hydrogen", "Helium"],
            "answer": "0",
            "image": "image1.jpg",
        }
    }
    with patch("json.load", return_value=ann_content):
        with patch("os.listdir", return_value=["1"]):
            with patch("os.path.join", return_value="path/to/image.jpg"):
                with patch(
                    "builtins.open",
                    mock.mock_open(read_data=json.dumps(ann_content)),
                ):
                    examples = scienceqa_loader("dummy_path", args)
                    assert len(examples) == 1
                    assert examples[0]["label"] == 0
                    assert examples[0]["image_path"] == "path/to/image.jpg"


def test_ai2d_loader(sample_args):
    args = sample_args
    args.num_options = 3
    question_content = {
        "questions": {
            "What is this?": {
                "answerTexts": ["Cat", "Dog", "Mouse"],
                "correctAnswer": "1",
                "abcLabel": False,
            }
        },
        "imageName": "image1.jpg",
    }
    with patch("os.listdir", return_value=["file1.json"]):
        with patch("json.load", return_value=question_content):
            with patch(
                "os.path.join",
                side_effect=lambda *args: "path/to/" + "/".join(args[-2:]),
            ):
                with patch(
                    "builtins.open",
                    mock.mock_open(read_data=json.dumps(question_content)),
                ):
                    examples = ai2d_loader("dummy_path", args)
                    assert len(examples) == 1
                    assert examples[0]["label"] == 1
                    assert (
                        examples[0]["image_path"]
                        == "path/to/dummy_path/ai2d/images/image1.jpg"
                    )


# New test functions to increase coverage


def test_preprocess_function_causal_vqa(sample_args):
    examples = {
        "question": ["What is shown in the image?"],
        "choice0": ["Cat"],
        "choice1": ["Dog"],
        "image_path": ["path/to/image1.jpg"],
    }
    processor = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    # Adjust the tokenizer to return lists of lists
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }
    processor.tokenizer = tokenizer
    data_obj = MagicMock()
    data_obj.data = {
        "pixel_values": torch.tensor(
            [[[1, 2], [3, 4]]] * 2
        )  # Repeat to match the number of choices
    }
    processor.image_processor.return_value = data_obj
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "image_header_name": "image_path",
        "processor": processor,
        "image_token": "",
    }
    with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
        output = preprocess_function_causal_vqa(examples, **kwargs)
        assert "input_ids" in output
        assert "labels" in output
        assert "header_attention_mask" in output
        assert "ending_attention_mask" in output
        assert "images" in output


def test_preprocess_function_seq2seq_vqa_channel(sample_args):
    examples = {
        "question": ["What is shown in the image?"],
        "choice0": ["Cat"],
        "choice1": ["Dog"],
        "image_path": ["path/to/image1.jpg"],
    }
    processor = MagicMock()
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }
    processor.tokenizer = tokenizer
    data_obj = MagicMock()
    data_obj.data = {
        "pixel_values": torch.tensor(
            [[[1, 2], [3, 4]]] * 2
        )  # Repeat to match the number of choices
    }
    processor.image_processor.return_value = data_obj
    kwargs = {
        "ending_names": ["choice0", "choice1"],
        "header_name": "question",
        "image_header_name": "image_path",
        "processor": processor,
        "image_token": "",
    }
    with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
        output = preprocess_function_seq2seq_vqa_channel(examples, **kwargs)
        assert "header_input_ids" in output
        assert "ending_input_ids" in output
        assert "images" in output
        assert len(output["images"]) == len(examples["question"])
        for img_list in output["images"]:
            assert len(img_list) == len(kwargs["ending_names"])


def test_preprocess_function_causal_vqa_channel(sample_args):
    examples = {
        "question": ["What is shown in the image?"],
        "hypothesis0": ["Cat"],
        "hypothesis1": ["Dog"],
        "image_path": ["path/to/image1.jpg"],
    }
    processor = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.return_value = {
        "input_ids": [[1, 2], [3, 4]],
        "attention_mask": [[1, 1], [1, 1]],
    }
    processor.tokenizer = tokenizer
    data_obj = MagicMock()
    data_obj.data = {"pixel_values": torch.tensor([[[1, 2], [3, 4]]] * 2)}
    processor.image_processor.return_value = data_obj
    kwargs = {
        "ending_names": ["hypothesis0", "hypothesis1"],
        "header_name": "question",
        "image_header_name": "image_path",
        "processor": processor,
        "image_token": "",
    }
    with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
        output = preprocess_function_causal_vqa_channel(examples, **kwargs)
        assert "input_ids" in output
        assert "labels" in output
        assert "header_attention_mask" in output
        assert "ending_attention_mask" in output
        assert "images" in output


def test_piqa_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    qa_content = json.dumps(
        {
            "goal": "To open a jar, you should",
            "sol1": "Twist the lid counter-clockwise",
            "sol2": "Push the lid upwards",
        }
    )
    label_content = "0\n"  # First solution is correct
    with patch("builtins.open", mock.mock_open()) as mock_file:
        mock_file.side_effect = [
            mock.mock_open(read_data=qa_content).return_value,
            mock.mock_open(read_data=label_content).return_value,
        ]
        examples = piqa_loader(
            ["dummy_qa_path.jsonl", "dummy_label_path.txt"], args
        )
        assert len(examples) == 1
        assert examples[0]["label"] == 0
        assert (
            "Answer the following question: Question: To open a jar, you should"
            in examples[0]["premise"]
        )


def test_qasc_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    json_line = json.dumps(
        {
            "answerKey": "B",
            "question": {
                "stem": "What do plants need to perform photosynthesis?",
                "choices": [
                    {"label": "A", "text": "Oxygen"},
                    {"label": "B", "text": "Sunlight"},
                    {"label": "C", "text": "Nitrogen"},
                    {"label": "D", "text": "Carbon dioxide"},
                    {"label": "E", "text": "Water"},
                    {"label": "F", "text": "Soil"},
                    {"label": "G", "text": "Minerals"},
                    {"label": "H", "text": "Glucose"},
                ],
            },
        }
    )
    with patch("builtins.open", mock.mock_open(read_data=json_line)):
        examples = qasc_loader("dummy_path.jsonl", args)
        assert len(examples) == 1
        assert examples[0]["label"] == 1  # 'B' corresponds to index 1
        assert (
            "Answer the following question: Question: What do plants need to perform photosynthesis?"
            in examples[0]["premise"]
        )


def test_siqa_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    qa_content = json.dumps(
        {
            "context": "Alex went to the store.",
            "question": "Why did Alex go to the store?",
            "answerA": "To buy groceries",
            "answerB": "To sell groceries",
            "answerC": "To sleep",
        }
    )
    label_content = "1\n"  # Answer index is 1 (but labels are 1-based in siqa_loader, and subtract 1)
    with patch("builtins.open", mock.mock_open()) as mock_file:
        mock_file.side_effect = [
            mock.mock_open(read_data=qa_content).return_value,
            mock.mock_open(read_data=label_content).return_value,
        ]
        examples = siqa_loader(
            ["dummy_qa_path.jsonl", "dummy_label_path.txt"], args
        )
        assert len(examples) == 1
        assert (
            examples[0]["label"] == 0
        )  # '1' in label file corresponds to index 0
        assert (
            "Answer the following question: Question: Alex went to the store. Why did Alex go to the store?"
            in examples[0]["premise"]
        )


def test_winogrande_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    qa_content = json.dumps(
        {
            "sentence": "The trophy doesn't fit in the brown suitcase because it's too big.",
            "option1": "trophy",
            "option2": "suitcase",
        }
    )
    label_content = "1\n"  # Correct answer is option1 (labels are 1-based)
    with patch("builtins.open", mock.mock_open()) as mock_file:
        mock_file.side_effect = [
            mock.mock_open(read_data=qa_content).return_value,
            mock.mock_open(read_data=label_content).return_value,
        ]
        examples = winogrande_loader(
            ["dummy_qa_path.jsonl", "dummy_label_path.txt"], args
        )
        assert len(examples) == 1
        assert (
            examples[0]["label"] == 0
        )  # '1' in label file corresponds to index 0
        assert (
            "Answer the following question: Question: The trophy doesn't fit in the brown suitcase because it's too big."
            in examples[0]["premise"]
        )


def test_date_understanding_loader(sample_args):
    args = sample_args
    args.multiple_choice_prompt = "Answer the following question:"
    args.num_options = 2
    data_content = {
        "task_prefix": "",
        "examples": [
            {"input": "What is 2+2?", "target_scores": {"4": 1, "5": 0}}
        ],
    }
    with patch("json.load", return_value=data_content):
        with patch(
            "builtins.open", mock.mock_open(read_data=json.dumps(data_content))
        ):
            examples = date_understanding_loader(["dummy_path.json"], args)
            assert len(examples) == 1
            assert examples[0]["label"] == 0  # '4' is at index 0
            assert (
                "Answer the following question: Question: What is 2+2?"
                in examples[0]["premise"]
            )


# Now this test_data.py covers only 60% of all tests when I run pytest coverage. You need to add more tests to this file. DO NOT CHANGE ANYTHING WRITTEN IN THIS FILE, NO CHANGE TO ANY OF THE CURRENT TESTS, ALL OF THEM ARE WORKING. ADD NEW TESTS TO THIS FILE TO GET 100% COVERAGE
