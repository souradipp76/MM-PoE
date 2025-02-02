from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GenerationConfig


def inference_language_modeling_old(model, eval_dataloader, device):
    model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
        ending_shape = batch["ending_input_ids"].shape
        # flatten
        header_input_ids = (
            batch["header_input_ids"]
            .view(-1, batch["header_input_ids"].shape[-1])
            .to(device)
        )
        ending_input_ids = (
            batch["ending_input_ids"]
            .view(-1, batch["ending_input_ids"].shape[-1])
            .to(device)
        )

        # adding this line of code takes me more than an hour.
        # without adding torch.no_grad, GPU usage will muiltply by 4.
        with torch.no_grad():
            outputs = model(
                input_ids=header_input_ids, labels=ending_input_ids
            )

        _, logits = outputs.loss, outputs.logits
        # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
        logits = logits.view(-1, logits.shape[-1])
        # ignore padding token: 0
        ce_loss = (
            F.cross_entropy(
                logits,
                ending_input_ids.view(-1),
                reduction="none",
                ignore_index=0,
            )
            .detach()
            .cpu()
        )
        # each score is the negative log-likelihood of a ending given a header.
        batch_predictions = (
            ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
        )
        batch_labels = batch["label"]
        predictions = torch.cat((predictions, batch_predictions))
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        batch_accuracy = (
            batch_predictions == batch_labels
        ).sum().item() / len(batch_labels)
        total_accuracy = (predictions == labels).sum().item() / len(labels)
        pbar.set_description(
            f"Total Accuracy: {total_accuracy:.4f}, "
            + f"Batch Accuracy: {batch_accuracy:.4f}"
        )
    return total_accuracy


def inference_contrastive_decoding_old(
    amateur_model, expert_model, eval_dataloader, device
):
    amateur_model.eval()
    expert_model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
        ending_shape = batch["ending_input_ids"].shape
        # flatten
        header_input_ids = (
            batch["header_input_ids"]
            .view(-1, batch["header_input_ids"].shape[-1])
            .to(device)
        )
        ending_input_ids = (
            batch["ending_input_ids"]
            .view(-1, batch["ending_input_ids"].shape[-1])
            .to(device)
        )

        # key step: compute logits.
        with torch.no_grad():
            amateur_model_logits = amateur_model(
                input_ids=header_input_ids, labels=ending_input_ids
            ).logits
            expert_model_logits = expert_model(
                input_ids=header_input_ids, labels=ending_input_ids
            ).logits

        logits = expert_model_logits - amateur_model_logits
        # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
        logits = logits.view(-1, logits.shape[-1])
        # ignore padding token: 0
        ce_loss = (
            F.cross_entropy(
                logits,
                ending_input_ids.view(-1),
                reduction="none",
                ignore_index=0,
            )
            .detach()
            .cpu()
        )
        # each score is the negative log-likelihood of a ending given a header.
        batch_predictions = (
            ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
        )
        batch_labels = batch["label"]
        predictions = torch.cat((predictions, batch_predictions))
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        batch_accuracy = (
            batch_predictions == batch_labels
        ).sum().item() / len(batch_labels)
        total_accuracy = (predictions == labels).sum().item() / len(labels)
        pbar.set_description(
            f"Total Accuracy: {total_accuracy:.4f}, "
            + f"Batch Accuracy: {batch_accuracy:.4f}"
        )
    return total_accuracy


def inference_language_modeling(
    model, eval_dataloader, device, compute_func, pad_token_id
):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions)
        )
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(
            labels
        )
        pbar.set_description(
            f"Language modeling accuracy: {lm_accuracy:.4f}, "
            + f"Average language modeling accuracy: {avg_lm_accuracy:.4f}"
        )
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions


def inference_generate_synonyms(
    model,
    eval_dataloader,
    device,
    compute_func,
    pad_token_id,
    num_of_options,
    num_of_synonyms,
):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        # need to aggregate according to original options.
        # each row in log_prob correspond to: h0, h1, ..., hn,
        # h0s0, h0s1, ..., h1s0, ...,

        # indexing log_prob to rearrange rows by keeping
        # options with corresponding synonyms together.
        log_prob = aggregate_optionw_with_synonyms(
            log_prob, num_of_options, num_of_synonyms
        )
        avg_log_prob = aggregate_optionw_with_synonyms(
            avg_log_prob, num_of_options, num_of_synonyms
        )
        # then reshape, and then aggregate options and synonyms by averaging.
        log_prob = log_prob.view(-1, num_of_options, num_of_synonyms + 1).mean(
            dim=-1
        )
        avg_log_prob = avg_log_prob.view(
            -1, num_of_options, num_of_synonyms + 1
        ).mean(dim=-1)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions)
        )
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(
            labels
        )
        pbar.set_description(
            f"Language modeling accuracy: {lm_accuracy:.4f}, "
            + f"Average language modeling accuracy: {avg_lm_accuracy:.4f}"
        )
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions


def inference_calibration(
    model,
    eval_dataloader,
    eval_calibration_dataloader,
    device,
    compute_func,
    pad_token_id,
):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(
        zip(eval_dataloader, eval_calibration_dataloader),
        desc="Inference",
        total=len(eval_dataloader),
    )
    for batch, batch_calibration in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        log_prob_calibration = compute_func(
            batch_calibration, model, device, pad_token_id
        )
        log_prob = log_prob - log_prob_calibration
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions)
        )
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(
            labels
        )
        pbar.set_description(
            f"Calibration accuracy: {lm_accuracy:.4f}, "
            + f"Average calibration accuracy: {avg_lm_accuracy:.4f}"
        )
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions


def inference_contrastive_decoding(method, model, **kwargs):
    args = kwargs["args"]
    raw_dataset = kwargs["raw_dataset"]
    device = kwargs["device"]
    compute_func = kwargs["compute_func"]
    tokenizer = kwargs["tokenizer"]
    processor = kwargs["processor"]
    ending_names = kwargs["ending_names"]
    header_name = kwargs["header_name"]
    image_header_name = kwargs["image_header_name"]
    preprocess_func = kwargs["preprocess_func"]
    preprocess_func_channel = kwargs["preprocess_func_channel"]

    fn_kwargs = {
        "ending_names": ending_names,
        "header_name": header_name,
        "tokenizer": tokenizer,
    }
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
        }
    # num_of_options = len(ending_names)
    tokenized_dataset = raw_dataset.map(
        preprocess_func,
        fn_kwargs=fn_kwargs,
        batched=True,
        batch_size=args.batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=False
    )
    if method in ["language_modeling", "multiple_choice_prompt"]:
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_language_modeling(
                model,
                eval_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        )
    elif method == "calibration":
        fn_kwargs = {
            "ending_names": ending_names,
            "header_name": "uncond_premise",  # the difference is here
            "tokenizer": tokenizer,
        }
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
                "header_name": "uncond_premise",
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
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_calibration(
                model,
                eval_dataloader,
                eval_calibration_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        )
    elif method == "channel":
        # simple solution: swap first sentence and
        # second sentence in both preprocessing functions
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
        avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions = (
            inference_language_modeling(
                model,
                eval_channel_dataloader,
                device,
                compute_func,
                tokenizer.pad_token_id,
            )
        )
    else:
        raise NotImplementedError
    return avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions


def compute_mask_process_of_elimination(
    avg_log_probs, mask_strategy, **kwargs
):
    masks = torch.ones_like(avg_log_probs)
    if mask_strategy == "lowest":
        # soft masking (v1), i.e., get rid of the least likely answer.
        masks[
            torch.arange(avg_log_probs.shape[0]), avg_log_probs.argmax(dim=-1)
        ] = 0
    elif mask_strategy == "below_average":
        # v2: Calculate the row-wise mean
        row_mean = avg_log_probs.mean(dim=1, keepdim=True)
        # Set values below the mean to 0
        masks[avg_log_probs > row_mean] = 0
    elif mask_strategy == "lowest_iter":
        # similar to lowest, but ignore inf,
        # and mask from the remaining options.
        # soft masking (v1), i.e., get rid of the least likely answer.
        avg_log_probs[avg_log_probs == float("inf")] = float("-inf")
        masks[
            torch.arange(avg_log_probs.shape[0]), avg_log_probs.argmax(dim=-1)
        ] = 0
        # set mask that correspond to inf to 0
        masks[avg_log_probs == float("-inf")] = 0
    elif mask_strategy == "min_k":
        min_k = kwargs["min_k"]
        # keep the min k options
        avg_log_probs_f32 = avg_log_probs.float()
        _, min_k_indices = avg_log_probs_f32.topk(min_k, dim=-1)
        masks[
            torch.arange(avg_log_probs_f32.shape[0]).unsqueeze(-1),
            min_k_indices,
        ] = 0
    else:
        raise NotImplementedError
    return masks


def inference_process_of_elimination(
    model, eval_dataloader, device, compute_func, pad_token_id
):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        # apply hard masking
        log_prob[batch["mask"] == 0] = float("inf")
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat(
            (avg_lm_predictions, batch_avg_predictions)
        )
        labels = torch.cat((labels, batch_labels))

        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(
            labels
        )
        pbar.set_description(
            f"Process of elimination accuracy: {lm_accuracy:.4f}, "
            + f"Average process of elimination accuracy: {avg_lm_accuracy:.4f}"
        )
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy, lm_predictions


def compute_conditional_score_seq2seq(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch

    # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
    ending_shape = batch["ending_input_ids"].shape
    # flatten. both input_ids has 0 as padding token.
    header_input_ids = (
        batch["header_input_ids"]
        .view(-1, batch["header_input_ids"].shape[-1])
        .to(device)
    )
    header_attention_mask = (
        batch["header_attention_mask"]
        .view(-1, batch["header_attention_mask"].shape[-1])
        .to(device)
    )
    ending_input_ids = (
        batch["ending_input_ids"]
        .view(-1, batch["ending_input_ids"].shape[-1])
        .to(device)
    )

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(
            input_ids=header_input_ids,
            attention_mask=header_attention_mask,
            labels=ending_input_ids,
        )

    _, logits = outputs.loss, outputs.logits
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 0
    ce_loss = (
        F.cross_entropy(
            logits,
            ending_input_ids.view(-1),
            reduction="none",
            ignore_index=pad_token_id,
        )
        .detach()
        .cpu()
    )
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    log_prob = ce_loss.view(ending_shape).sum(dim=-1)
    return log_prob


def compute_conditional_score_causal(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch
    # make sure the padding token is aligned with tokenizer.pad_token_id
    # and preprocess_function_causal
    # padding_token = 50256

    input_ids = (
        batch["input_ids"].view(-1, batch["input_ids"].shape[-1]).to(device)
    )
    labels = batch["labels"].view(-1, batch["labels"].shape[-1]).to(device)

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            # attention_mask = attention_mask,
            labels=labels,
        )

    _, logits = outputs.loss, outputs.logits
    # shift
    logits = logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 50256
    ce_loss = (
        F.cross_entropy(
            logits,
            labels.view(-1),
            reduction="none",
            ignore_index=pad_token_id,
        )
        .detach()
        .cpu()
    )
    # each score is the negative log-likelihood of a ending given a header.
    log_prob = ce_loss.view(
        batch["input_ids"].shape[0], batch["input_ids"].shape[1], -1
    ).sum(dim=-1)
    return log_prob


def compute_conditional_score_seq2seq_vqa(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch

    # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
    ending_shape = batch["ending_input_ids"].shape
    # flatten. both input_ids has 0 as padding token.
    header_input_ids = (
        batch["header_input_ids"]
        .view(-1, batch["header_input_ids"].shape[-1])
        .to(device)
    )
    header_attention_mask = (
        batch["header_attention_mask"]
        .view(-1, batch["header_attention_mask"].shape[-1])
        .to(device)
    )
    ending_input_ids = (
        batch["ending_input_ids"]
        .view(-1, batch["ending_input_ids"].shape[-1])
        .to(device)
    )
    images = (
        batch["images"]
        .view(
            -1,
            batch["images"].shape[-3],
            batch["images"].shape[-2],
            batch["images"].shape[-1],
        )
        .to(device)
    )

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(
            input_ids=header_input_ids,
            attention_mask=header_attention_mask,
            pixel_values=images,
            labels=ending_input_ids,
        )

    _, logits = outputs.loss, outputs.logits
    logits = logits[:, : ending_input_ids.shape[-1], :]
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.contiguous().view(-1, logits.shape[-1])
    # ignore padding token: 0
    ce_loss = (
        F.cross_entropy(
            logits,
            ending_input_ids.view(-1),
            reduction="none",
            ignore_index=pad_token_id,
        )
        .detach()
        .cpu()
    )
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    log_prob = ce_loss.view(ending_shape).sum(dim=-1)
    return log_prob


def compute_conditional_score_causal_vqa(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch
    # make sure the padding token is aligned with tokenizer.pad_token_id
    # and preprocess_function_causal
    # padding_token = 50256
    input_ids = (
        batch["input_ids"].view(-1, batch["input_ids"].shape[-1]).to(device)
    )
    header_attention_mask = (
        batch["header_attention_mask"]
        .view(-1, batch["header_attention_mask"].shape[-1])
        .to(device)
    )
    # ending_attention_mask = (
    #     batch["ending_attention_mask"]
    #     .view(-1, batch["ending_attention_mask"].shape[-1])
    #     .to(device)
    # )
    labels = batch["labels"].view(-1, batch["labels"].shape[-1]).to(device)
    images = (
        batch["images"]
        .view(
            -1,
            batch["images"].shape[-3],
            batch["images"].shape[-2],
            batch["images"].shape[-1],
        )
        .to(device)
    )

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=images,
            attention_mask=header_attention_mask,
            labels=labels,
        )

    _, logits = outputs.loss, outputs.logits
    logits = logits[:, -labels.shape[-1] :, :]  # for GIT

    # shift
    logits = logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()

    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 50256
    ce_loss = (
        F.cross_entropy(
            logits,
            labels.view(-1),
            reduction="none",
            ignore_index=pad_token_id,
        )
        .detach()
        .cpu()
    )
    # each score is the negative log-likelihood of a ending given a header.
    log_prob = ce_loss.view(
        batch["input_ids"].shape[0], batch["input_ids"].shape[1], -1
    ).sum(dim=-1)
    return log_prob


def generate_synonyms(args, model, tokenizer, tokenized_dataset):

    generation_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=args.number_of_synonyms,
    )

    # get all columns of tokenized_dataset that starts with "hypothesis"
    hypothesis_columns = [
        col
        for col in tokenized_dataset.column_names
        if col.startswith("hypothesis")
    ]
    # batch inference? May check SEQA code or HF doc.
    synonyms_dict = {}
    for col in tqdm(hypothesis_columns, desc="Generate synonyms"):
        for option in tqdm(
            tokenized_dataset[col], desc=f"Generate synonyms for {col}"
        ):
            # prompt = f"Generate a synonym to '{option}':"
            prompt = args.generate_synonyms_prompt.replace(
                "'{option}'", option
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, generation_config=generation_config
            )
            synonyms = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            # store the synonyms, so map() is easy.
            # directly mapping here? All I need to do
            # is to create duplicates of the instance with synonyms.
            synonyms_dict[option] = synonyms

    return synonyms_dict


def aggregate_optionw_with_synonyms(tensor, num_of_options, num_of_synonyms):
    # this function changes the column order.
    # tensor: (batch_size, num_of_options * (num_of_synonyms + 1))
    old_index = list(range(tensor.shape[1]))
    aggregated_index = [-1] * len(old_index)
    # exapmle: commonsenseqa 5 options with 3 synonyms: 0, 4, 8, 12, 16
    options_index_old = list(range(num_of_options))  # e.g., 0..4
    options_index_new = [
        i * (num_of_synonyms + 1) for i in options_index_old
    ]  # e.g., 0, 4, 8, 12, 16
    remain_index = [
        i for i in old_index if i not in options_index_old
    ]  # e.g., 5..19
    for i, _ in enumerate(aggregated_index):
        if i in options_index_new:  # 0, 4, 8, 12, 16
            aggregated_index[i] = options_index_old.pop(0)
        else:
            aggregated_index[i] = remain_index.pop(0)

    # aggregated_index = options_index + \
    #     [i for i in old_index if i not in options_index]
    tensor[:, old_index] = tensor[:, aggregated_index]
    return tensor
