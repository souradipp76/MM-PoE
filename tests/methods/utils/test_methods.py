import os
import sys
import torch
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the functions from methods.py
from mm_poe.methods.utils.methods import (
    inference_language_modeling_old,
    inference_contrastive_decoding_old,
    inference_language_modeling,
    inference_generate_synonyms,
    inference_calibration,
    inference_contrastive_decoding,
    compute_mask_process_of_elimination,
    inference_process_of_elimination,
    compute_conditional_score_seq2seq,
    compute_conditional_score_causal,
    compute_conditional_score_seq2seq_vqa,
    compute_conditional_score_causal_vqa,
    generate_synonyms,
    aggregate_optionw_with_synonyms,
)

# Mock tqdm to prevent actual progress bars during testing
tqdm = lambda x, **kwargs: x


# Define a simple mock model
class SimpleMockModel(torch.nn.Module):
    def __init__(self):
        super(SimpleMockModel, self).__init__()

    def forward(
        self, input_ids=None, labels=None, pixel_values=None, **kwargs
    ):
        seq_len = input_ids.size(1)
        batch_size_num_options = labels.size(0)
        vocab_size = 32128
        logits = torch.randn(batch_size_num_options, seq_len, vocab_size)
        loss = torch.tensor(0.0)
        return MagicMock(loss=loss, logits=logits)


# Fixtures for common test components
@pytest.fixture
def mock_model():
    return SimpleMockModel()


@pytest.fixture
def mock_amateur_model():
    return SimpleMockModel()


@pytest.fixture
def mock_expert_model():
    return SimpleMockModel()


@pytest.fixture
def sample_batch():
    batch_size = 2
    num_options = 2
    seq_len = 18
    vocab_size = 32128
    return {
        "ending_input_ids": torch.randint(
            0, vocab_size, (batch_size, num_options, seq_len)
        ),
        "header_input_ids": torch.randint(
            0, vocab_size, (batch_size, seq_len)
        ),
        "label": torch.randint(0, num_options, (batch_size,)),
        "header_attention_mask": torch.ones(batch_size, seq_len),
        "ending_attention_mask": torch.ones(batch_size, num_options, seq_len),
        "input_ids": torch.randint(
            0, vocab_size, (batch_size, num_options, seq_len)
        ),
        "labels": torch.randint(
            0, vocab_size, (batch_size, num_options, seq_len)
        ),
        "mask": torch.ones(batch_size, num_options),
        "images": torch.randn(batch_size, 3, 224, 224),
    }


@pytest.fixture
def sample_synonym_batch():
    batch_size = 2
    num_synonyms = 2
    num_options = 2
    seq_len = 18
    vocab_size = 32128
    return {
        "ending_input_ids": torch.randint(
            0,
            vocab_size,
            (batch_size, num_options * (num_synonyms + 1), seq_len),
        ),
        "header_input_ids": torch.randint(
            0, vocab_size, (batch_size, seq_len)
        ),
        "label": torch.randint(
            0, num_options * (num_synonyms + 1), (batch_size,)
        ),
        "header_attention_mask": torch.ones(batch_size, seq_len),
        "ending_attention_mask": torch.ones(
            batch_size, num_options * (num_synonyms + 1), seq_len
        ),
        "input_ids": torch.randint(
            0,
            vocab_size,
            (batch_size, num_options * (num_synonyms + 1), seq_len),
        ),
        "labels": torch.randint(
            0,
            vocab_size,
            (batch_size, num_options * (num_synonyms + 1), seq_len),
        ),
        "mask": torch.ones(batch_size, num_options * (num_synonyms + 1)),
        "images": torch.randn(batch_size, 3, 224, 224),
    }


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def pad_token_id():
    return 0


# Tests for inference_language_modeling_old function
def test_inference_language_modeling_old(mock_model, sample_batch, device):
    eval_dataloader = [sample_batch]
    total_accuracy = inference_language_modeling_old(
        mock_model, eval_dataloader, device
    )
    assert isinstance(total_accuracy, float)
    assert 0.0 <= total_accuracy <= 1.0


# Tests for inference_contrastive_decoding_old function
def test_inference_contrastive_decoding_old(
    mock_amateur_model, mock_expert_model, sample_batch, device
):
    eval_dataloader = [sample_batch]
    total_accuracy = inference_contrastive_decoding_old(
        mock_amateur_model, mock_expert_model, eval_dataloader, device
    )
    assert isinstance(total_accuracy, float)
    assert 0.0 <= total_accuracy <= 1.0


# Mock compute_func for inference_language_modeling
def mock_compute_func(batch, model, device, pad_token_id):
    batch_size = batch["header_input_ids"].size(0)
    num_options = batch["ending_input_ids"].size(1)
    return torch.rand(batch_size, num_options)


# Tests for inference_language_modeling function
def test_inference_language_modeling(
    mock_model, sample_batch, device, pad_token_id
):
    eval_dataloader = [sample_batch]
    avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
        inference_language_modeling(
            mock_model,
            eval_dataloader,
            device,
            mock_compute_func,
            pad_token_id,
        )
    )
    assert avg_log_probs.shape == (
        sample_batch["label"].size(0),
        sample_batch["ending_input_ids"].size(1),
    )
    assert isinstance(lm_accuracy, float)
    assert isinstance(avg_lm_accuracy, float)
    assert lm_predictions.shape == (sample_batch["label"].size(0),)


# Tests for inference_generate_synonyms function
def test_inference_generate_synonyms(
    mock_model, sample_synonym_batch, device, pad_token_id
):
    num_of_options = 2
    num_of_synonyms = 2

    def mock_compute_func(batch, model, device, pad_token_id):
        batch_size = batch["header_input_ids"].size(0)
        total_options = batch["ending_input_ids"].size(1)
        return torch.rand(batch_size, total_options)

    eval_dataloader = [sample_synonym_batch]
    avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
        inference_generate_synonyms(
            mock_model,
            eval_dataloader,
            device,
            mock_compute_func,
            pad_token_id,
            num_of_options,
            num_of_synonyms,
        )
    )
    expected_shape = (
        sample_synonym_batch["label"].size(0),
        num_of_options * (num_of_synonyms + 1),
    )
    assert avg_log_probs.shape == expected_shape
    assert isinstance(lm_accuracy, float)
    assert isinstance(avg_lm_accuracy, float)
    assert lm_predictions.shape == (sample_synonym_batch["label"].size(0),)


# Tests for inference_calibration function
def test_inference_calibration(mock_model, sample_batch, device, pad_token_id):
    eval_dataloader = [sample_batch]
    eval_calibration_dataloader = [sample_batch]
    avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
        inference_calibration(
            mock_model,
            eval_dataloader,
            eval_calibration_dataloader,
            device,
            mock_compute_func,
            pad_token_id,
        )
    )
    assert avg_log_probs.shape == (
        sample_batch["label"].size(0),
        sample_batch["ending_input_ids"].size(1),
    )
    assert isinstance(lm_accuracy, float)
    assert isinstance(avg_lm_accuracy, float)
    assert lm_predictions.shape == (sample_batch["label"].size(0),)


# Tests for compute_mask_process_of_elimination function
@pytest.mark.parametrize(
    "mask_strategy", ["lowest", "below_average", "lowest_iter", "min_k"]
)
def test_compute_mask_process_of_elimination(mask_strategy):
    avg_log_probs = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    if mask_strategy == "min_k":
        kwargs = {"min_k": 2}
    else:
        kwargs = {}
    if mask_strategy not in [
        "lowest",
        "below_average",
        "lowest_iter",
        "min_k",
    ]:
        with pytest.raises(NotImplementedError):
            compute_mask_process_of_elimination(
                avg_log_probs, mask_strategy, **kwargs
            )
    else:
        masks = compute_mask_process_of_elimination(
            avg_log_probs, mask_strategy, **kwargs
        )
        assert masks.shape == avg_log_probs.shape


# Tests for inference_process_of_elimination function
def test_inference_process_of_elimination(
    mock_model, sample_batch, device, pad_token_id
):
    eval_dataloader = [sample_batch]
    avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
        inference_process_of_elimination(
            mock_model,
            eval_dataloader,
            device,
            mock_compute_func,
            pad_token_id,
        )
    )
    assert avg_log_probs.shape == (
        sample_batch["label"].size(0),
        sample_batch["ending_input_ids"].size(1),
    )
    assert isinstance(lm_accuracy, float)
    assert isinstance(avg_lm_accuracy, float)
    assert lm_predictions.shape == (sample_batch["label"].size(0),)


# Tests for compute_conditional_score_seq2seq function
def test_compute_conditional_score_seq2seq(
    mock_model, sample_batch, device, pad_token_id
):
    log_prob = compute_conditional_score_seq2seq(
        sample_batch, mock_model, device, pad_token_id
    )
    assert log_prob.shape == (
        sample_batch["ending_input_ids"].shape[0],
        sample_batch["ending_input_ids"].shape[1],
    )


# Tests for compute_conditional_score_causal function
def test_compute_conditional_score_causal(
    mock_model, sample_batch, device, pad_token_id
):
    log_prob = compute_conditional_score_causal(
        sample_batch, mock_model, device, pad_token_id
    )
    assert log_prob.shape == (
        sample_batch["input_ids"].shape[0],
        sample_batch["input_ids"].shape[1],
    )


# Tests for compute_conditional_score_seq2seq_vqa function
def test_compute_conditional_score_seq2seq_vqa(
    mock_model, sample_batch, device, pad_token_id
):
    log_prob = compute_conditional_score_seq2seq_vqa(
        sample_batch, mock_model, device, pad_token_id
    )
    assert log_prob.shape == (
        sample_batch["ending_input_ids"].shape[0],
        sample_batch["ending_input_ids"].shape[1],
    )


# Tests for compute_conditional_score_causal_vqa function
def test_compute_conditional_score_causal_vqa(
    mock_model, sample_batch, device, pad_token_id
):
    log_prob = compute_conditional_score_causal_vqa(
        sample_batch, mock_model, device, pad_token_id
    )
    assert log_prob.shape == (
        sample_batch["input_ids"].shape[0],
        sample_batch["input_ids"].shape[1],
    )


# Tests for aggregate_optionw_with_synonyms function
def test_aggregate_optionw_with_synonyms():
    batch_size = 2
    num_of_options = 5
    num_of_synonyms = 3
    tensor = torch.arange(
        batch_size * num_of_options * (num_of_synonyms + 1)
    ).view(batch_size, -1)
    aggregated_tensor = aggregate_optionw_with_synonyms(
        tensor.clone(), num_of_options, num_of_synonyms
    )
    assert aggregated_tensor.shape == tensor.shape


# Tests for generate_synonyms function
def test_generate_synonyms():
    args = MagicMock()
    args.number_of_synonyms = 2
    args.generate_synonyms_prompt = "Generate a synonym to '{option}':"
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    tokenizer.return_tensors = "pt"
    tokenizer.pad_token_id = 0
    tokenizer.batch_decode.return_value = ["synonym1", "synonym2"]
    tokenized_dataset = MagicMock()
    tokenized_dataset.column_names = ["hypothesis1"]
    tokenized_dataset.__getitem__.return_value = {"hypothesis1": "test_option"}
    synonyms_dict = generate_synonyms(
        args, model, tokenizer, tokenized_dataset
    )
    assert isinstance(synonyms_dict, dict)


# Tests for inference_contrastive_decoding function
def test_inference_contrastive_decoding():
    method = "language_modeling"
    model = MagicMock()
    args = MagicMock()
    args.batch_size = 2
    args.model_family = "other"
    raw_dataset = MagicMock()
    device = "cpu"
    compute_func = MagicMock()
    tokenizer = MagicMock()
    processor = MagicMock()
    ending_names = ["ending1", "ending2"]
    header_name = "header"
    image_header_name = "image_header"
    preprocess_func = MagicMock()
    preprocess_func_channel = MagicMock()
    kwargs = {
        "args": args,
        "raw_dataset": raw_dataset,
        "device": device,
        "compute_func": compute_func,
        "tokenizer": tokenizer,
        "processor": processor,
        "ending_names": ending_names,
        "header_name": header_name,
        "image_header_name": image_header_name,
        "preprocess_func": preprocess_func,
        "preprocess_func_channel": preprocess_func_channel,
    }
    with patch(
        "mm_poe.methods.utils.methods.inference_language_modeling",
        return_value=(None, 0.0, 0.0, None),
    ) as mock_inference:
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_contrastive_decoding(method, model, **kwargs)
        )
        mock_inference.assert_called_once()

    method = "calibration"
    with patch(
        "mm_poe.methods.utils.methods.inference_calibration",
        return_value=(None, 0.0, 0.0, None),
    ) as mock_inference_cal:
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_contrastive_decoding(method, model, **kwargs)
        )
        mock_inference_cal.assert_called_once()

    method = "channel"
    with patch(
        "mm_poe.methods.utils.methods.inference_language_modeling",
        return_value=(None, 0.0, 0.0, None),
    ) as mock_inference_channel:
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_contrastive_decoding(method, model, **kwargs)
        )
        mock_inference_channel.assert_called()

    method = "invalid_method"
    with pytest.raises(NotImplementedError):
        inference_contrastive_decoding(method, model, **kwargs)
