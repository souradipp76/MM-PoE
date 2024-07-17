import json
import os
import xml.etree.ElementTree as ET
import random

import torch
from huggingface_hub import upload_folder
from PIL import Image

# write my own data loader, or using HF dataloader?
# steps for data loader: label, premise, options, hypothesis.
# uncond_premise = " the answer is:"

def upload_to_huggingface_hub(dataset, args):
    suffix = f"{args.dataset}_{args.seed}_{args.n_shot}_{args.sample}_{args.checkpoint.split('/')[-1]}_{args.batch_size}"
    temp_data_path = os.path.join(f"../temp_data/{args.method}", suffix)
    dataset.save_to_disk(temp_data_path)
    # _ = upload_folder(
    #     folder_path=temp_data_path, 
    #     path_in_repo=f"temp_data/{args.method}/{suffix}",
    #     repo_id="Vanmas/PoE_data",
    #     repo_type="dataset",)
    # remove the temp data folder
    # os.system(f"rm -rf {temp_data_path}")

def preprocess_function_seq2seq(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}

def preprocess_function_causal(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, truncation=True)
    tokenized_endings = tokenizer(second_sentences, truncation=True)

    # reference: https://github.com/peterwestuw/surface-form-competition/blob/main/utils.py#L177
    max_len = max(len(header + ending) for header, ending in zip(tokenized_headers['input_ids'], tokenized_endings['input_ids']))
    input_ids = torch.full((len(tokenized_headers['input_ids']), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = tokenizer.pad_token_id * torch.ones((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    ending_attention_mask = torch.zeros((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    for i, (header, ending) in enumerate(zip(tokenized_headers['input_ids'], tokenized_endings['input_ids'])):
        input_ids[i, :len(header)] = torch.tensor(header)
        input_ids[i, len(header):len(header)+len(ending)] = torch.tensor(ending)
        ending_attention_mask[i, len(header):len(header)+len(ending)] = torch.tensor(1)
        labels[i, len(header):len(header)+len(ending)] = torch.tensor(ending)

    flatten_dict = {"input_ids": input_ids, "labels": labels, "ending_attention_mask": ending_attention_mask}
    return_dict = {f"{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in flatten_dict.items()}
    return return_dict

def preprocess_function_vqa(examples, **kwargs):
    ending_names, header_name, image_header_name, processor = kwargs['ending_names'], kwargs['header_name'], kwargs['image_header_name'], kwargs['processor']
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    num_choice = len(ending_names)
    question_headers = examples[header_name]
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_headers = tokenizer(first_sentences, truncation=True)
    tokenized_endings = tokenizer(second_sentences, truncation=True)  

    image_paths = examples[image_header_name]
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    images = [[image] * len(ending_names) for image in images]
    images = sum(images, [])
    images = image_processor(images, return_tensors='pt').data

    max_len = max(len(header + ending) for header, ending in zip(tokenized_headers['input_ids'], tokenized_endings['input_ids']))
    input_ids = torch.full((len(tokenized_headers['input_ids']), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = tokenizer.pad_token_id * torch.ones((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    ending_attention_mask = torch.zeros((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    for i, (header, ending) in enumerate(zip(tokenized_headers['input_ids'], tokenized_endings['input_ids'])):
        input_ids[i, :len(header)] = torch.tensor(header)
        input_ids[i, len(header):len(header)+len(ending)] = torch.tensor(ending)
        ending_attention_mask[i, len(header):len(header)+len(ending)] = torch.tensor(1)
        labels[i, len(header):len(header)+len(ending)] = torch.tensor(ending)

    flatten_dict = {"input_ids": input_ids, "labels": labels, "ending_attention_mask": ending_attention_mask, "images": images["pixel_values"]}
    return_dict = {f"{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in flatten_dict.items()}
    return return_dict

def preprocess_function_seq2seq_channel(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # swap first_sentences and second_sentences
    first_sentences, second_sentences = second_sentences, first_sentences

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, padding=True, truncation=True)
    tokenized_endings = tokenizer(second_sentences, padding=True, truncation=True)
    header_dict = {f"header_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_headers.items()}
    ending_dict = {f"ending_{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_endings.items()}
    return {**header_dict, **ending_dict}

def preprocess_function_causal_channel(examples, **kwargs):
    ending_names, header_name, tokenizer = kwargs['ending_names'], kwargs['header_name'], kwargs['tokenizer']
    num_choice = len(ending_names)
    question_headers = examples[header_name]
    # the tokenizer handles multiple spaces.
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    # second_sentences = [
    #     [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_header)
    # ]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # swap first_sentences and second_sentences
    first_sentences, second_sentences = second_sentences, first_sentences

    # tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_headers = tokenizer(first_sentences, truncation=True)
    tokenized_endings = tokenizer(second_sentences, truncation=True)

    # reference: https://github.com/peterwestuw/surface-form-competition/blob/main/utils.py#L177
    max_len = max(len(header + ending) for header, ending in zip(tokenized_headers['input_ids'], tokenized_endings['input_ids']))
    input_ids = torch.full((len(tokenized_headers['input_ids']), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = tokenizer.pad_token_id * torch.ones((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    ending_attention_mask = torch.zeros((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    for i, (header, ending) in enumerate(zip(tokenized_headers['input_ids'], tokenized_endings['input_ids'])):
        input_ids[i, :len(header)] = torch.tensor(header)
        input_ids[i, len(header):len(header)+len(ending)] = torch.tensor(ending)
        ending_attention_mask[i, len(header):len(header)+len(ending)] = torch.tensor(1)
        labels[i, len(header):len(header)+len(ending)] = torch.tensor(ending)

    flatten_dict = {"input_ids": input_ids, "labels": labels, "ending_attention_mask": ending_attention_mask}
    return_dict = {f"{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in flatten_dict.items()}
    return return_dict

def preprocess_function_vqa_channel(examples, **kwargs):
    ending_names, header_name, image_header_name, processor = kwargs['ending_names'], kwargs['header_name'], kwargs['image_header_name'], kwargs['processor']
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    num_choice = len(ending_names)
    question_headers = examples[header_name]
    first_sentences = [[context] * len(ending_names) for context in examples[header_name]]
    second_sentences = [
        [f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

     # swap first_sentences and second_sentences
    first_sentences, second_sentences = second_sentences, first_sentences

    tokenized_headers = tokenizer(first_sentences, truncation=True)
    tokenized_endings = tokenizer(second_sentences, truncation=True)  

    image_paths = examples[image_header_name]
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    images = [[image] * len(ending_names) for image in images]
    images = sum(images, [])
    images = image_processor(images, return_tensors='pt').data

    max_len = max(len(header + ending) for header, ending in zip(tokenized_headers['input_ids'], tokenized_endings['input_ids']))
    input_ids = torch.full((len(tokenized_headers['input_ids']), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = tokenizer.pad_token_id * torch.ones((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    ending_attention_mask = torch.zeros((len(tokenized_headers['input_ids']), max_len), dtype=torch.long)
    for i, (header, ending) in enumerate(zip(tokenized_headers['input_ids'], tokenized_endings['input_ids'])):
        input_ids[i, :len(header)] = torch.tensor(header)
        input_ids[i, len(header):len(header)+len(ending)] = torch.tensor(ending)
        ending_attention_mask[i, len(header):len(header)+len(ending)] = torch.tensor(1)
        labels[i, len(header):len(header)+len(ending)] = torch.tensor(ending)

    flatten_dict = {"input_ids": input_ids, "labels": labels, "ending_attention_mask": ending_attention_mask, "images": images["pixel_values"]}
    return_dict = {f"{k}": [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in flatten_dict.items()}
    return return_dict

def create_multiple_choice_prompt(example, **kwargs):
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    multiple_choice_prompt = kwargs["multiple_choice_prompt"]
    scoring_method = kwargs["scoring_method"]
    num_of_options = kwargs["num_of_options"]
    mask = example['mask']
    # null_string = f"[MASK]"
    if kwargs['mask_token'] is not None:
        null_string = kwargs['mask_token']
    else:
        null_string = f"[MASK]"
    mcp_example = {}
    # example['premise'] = premise = f"{multiple_choice_prompt} {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
    # premise = f"{multiple_choice_prompt} Question: {example['premise']}\n"
    
    if scoring_method != "multiple_choice_prompt":
        premise = f"{multiple_choice_prompt}\n Question: {example['premise']}\n"
        premise = premise.replace(f"{example['uncond_premise']}", "")
        for idx, single_mask in enumerate(mask):
            mcp_example[f'hypothesis{idx}'] = alphabets[idx]
            if single_mask == 1:
                premise += f"{alphabets[idx]}. {example[f'hypothesis{idx}']}\n"
            else:
                # consider other null strings.
                premise += f"{alphabets[idx]}. {null_string}\n"
        premise += "Answer:"
    else: # for multiple choice prompt, options are already presented in the premise.
        premise = f"{multiple_choice_prompt}\n{example['premise']}"
        premise = premise.replace(f"{example['uncond_premise']}", "")
        for idx, single_mask in enumerate(mask):
            option_start_index = premise.rfind(f"{alphabets[idx]}. ")
            if idx == num_of_options - 1:
                option_end_index = premise.rfind(f"Answer:")
            else:
                option_end_index = premise.rfind(f"{alphabets[idx + 1]}. ")
            option = premise[option_start_index:option_end_index]
            if single_mask == 1:
                pass
            else:
                # consider other null strings.
                premise = premise.replace(option, f"{alphabets[idx]}. {null_string}\n")
    mcp_example['premise'] = premise
    return mcp_example

def create_synonym_dataset(examples, **kwargs):
    # for hypothesis0, create synonyms00, synonyms01, etc.
    args, synonyms_dict = kwargs['args'], kwargs["synonyms_dict"]
    number_of_synonyms = args.number_of_synonyms
    # get the hypothesis columns
    hypothesis_columns = [col for col in examples.keys() if "hypothesis" in col]
    for hypothesis_column in hypothesis_columns:
        for i in range(number_of_synonyms):
            examples[f"{hypothesis_column}_synonyms_{i}"] = [synonyms_dict[hypothesis][i] for hypothesis in examples[hypothesis_column]]
    return examples

def copa_loader(path, args):
    
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
        # examples_copa  += [{'options': [{'premise': ' ' + p[:-1] + bridge,
        #                                  'hypothesis': ' ' + a1,
        #                                  'uncond_premise': bridge,
        #                                  'uncond_hypothesis': ' ' + a1},
        #                                {'premise': ' ' + p[:-1] + bridge,
        #                                  'hypothesis': ' ' + a2,
        #                                  'uncond_premise': bridge,
        #                                  'uncond_hypothesis': ' ' + a2}], 
        #           'label':int(value)-1}]
        premise = ' ' + p[:-1] + bridge
        if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: The pond froze over for the winter so
                # A. People skated on the pond.
                # B. People brought boats to the pond.
                # Answer:
                hypotheses = ["A", "B"]
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {a1}\nB. {a2}\nAnswer:"
        else:
            hypotheses = [' ' + a1, ' ' + a2]
        examples_copa += [{
            'label': int(value)-1,
            'premise': premise,
            'uncond_premise': bridge,
            'hypothesis0': hypotheses[0],
            'hypothesis1': hypotheses[1],
        }]
    return examples_copa

def cqa_loader(path, args):
    examples_cqa = []
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' + d['question']['stem']
            premise = premise[:-1] + '?'  
            
            # to ensure identical perforamnce to the PMI paper.
            # options = [option['text'].lower() for option in d['question']['choices']]
            options = [f" \"{option['text'].lower()}\"" for option in d['question']['choices']]
            
            # examples += [{'options': [{'premise':premise + '? the answer is:' ,
                        #                 'hypothesis': ' "{}"'.format(c['text'].lower()),
                        #                 'uncond_premise': ' the answer is:',
                        #                 'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                        # 'label':label}]
            # if args.multiple_choice_prompt is not None:
            if getattr(args, 'multiple_choice_prompt', None) is not None:
                hypotheses = ["A", "B", "C", "D", "E"]
                # Question: How does a bishop move from one place to another?
                # A. chess game
                # B. church
                # C. in a car
                # D. queen
                # E. cathedral
                # Answer:
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nE. {options[4]}\nAnswer:"
            else:
                hypotheses = options
                premise = premise + uncond_premise
            examples_cqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
            }]
    return examples_cqa

def obqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    with open(path) as lines:
        abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }

        examples_obqa = []
        for line in lines:
            j = json.loads(line)

            label = abc2idx[j['answerKey']]
            premise = j['question']['stem']
            options_text = [f" {option['text']}" for option in j['question']['choices']]
            options_sym = [option['label'] for option in j['question']['choices']]
            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Greenhouses are great for plants like
                # A. Pizza
                # B. Lollipops
                # C. Candles
                # D. French beans
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nAnswer:"
            else:
                hypotheses = options_text
                # premise = premise + uncond_premise
            
            examples_obqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
            }]
    return examples_obqa

def piqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_piqa = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0])
            line = json.loads(line)
            premise = line['goal']
            options_text = [line['sol1'], line['sol2']]
            options_sym = ['A', 'B']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: To clear snot out of your nose,
                # A. place a tissue over your nose and blow the snot out.
                # B. place a tissue over your nose and suck the snot in.
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_piqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
            }]
    return examples_piqa

def qasc_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_qasc = []

    with open(path) as lines:
        for line in lines:
            line = json.loads(line)
            label = ['A','B','C','D','E','F','G','H'].index(line['answerKey'])
            premise = f"{line['question']['stem']}"

            options_text = [option['text'] for option in line['question']['choices']]
            options_sym = [option['label'] for option in line['question']['choices']]

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Cameron returned home with a bag of candy to eat all night
                # long. What will Others want to do next?
                # A. great
                # B. buy the candy to eat
                # C. bored
                # E. ...
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nF. {options_text[5]}\nG. {options_text[6]}\nH. {options_text[7]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_qasc += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
                'hypothesis3': hypotheses[3],
                'hypothesis4': hypotheses[4],
                'hypothesis5': hypotheses[5],
                'hypothesis6': hypotheses[6],
                'hypothesis7': hypotheses[7],
            }]
    return examples_qasc

def siqa_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_siqa = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0]) - 1
            line = json.loads(line)
            premise = f"{line['context']} {line['question']}"
            
            options_text = [line['answerA'], line['answerB'], line['answerC']]
            options_sym = ['A', 'B', 'C']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: Cameron returned home with a bag of candy to eat all night
                # long. What will Others want to do next?
                # A. great
                # B. buy the candy to eat
                # C. bored
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_siqa += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
                'hypothesis2': hypotheses[2],
            }]
    return examples_siqa

def winogrande_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples_winogrande = []
    qa_path, label_path = path[0], path[1]

    with open(qa_path) as lines, open(label_path) as labels:
        for line, label_sym in zip(lines, labels):
            label = int(label_sym[0]) - 1
            line = json.loads(line)
            premise = line['sentence']
            
            options_text = [line['option1'], line['option2']]
            options_sym = ['A', 'B']

            if getattr(args, 'multiple_choice_prompt', None) is not None:
                # Question: So _ plays video games because Leslie has a lot of free time
                # while Nelson has to work all the time.
                # A. Leslie
                # B. Nelson
                # Answer:
                hypotheses = options_sym
                premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nAnswer:"
            else:
                hypotheses = options_text
                premise = premise + uncond_premise
            
            examples_winogrande += [{
                'label': label,
                'premise': premise,
                'uncond_premise': uncond_premise,
                'hypothesis0': hypotheses[0],
                'hypothesis1': hypotheses[1],
            }]
    return examples_winogrande

def date_understanding_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []

    for one_path in path:
        with open(one_path) as json_file:
            data = json.load(json_file)
            task_prefix = data.get("task_prefix", "")
            for instance in data['examples']:
                options_text = list(instance['target_scores'].keys())
                num_options = len(options_text)
                if args.num_options is not None and num_options != args.num_options:
                    continue
                options_sym = [chr(ord('A') + i) for i in range(num_options)]
                for target, score in instance['target_scores'].items():
                    if score == 1:
                        raw_label = target # e.g., stare wars
                label = options_text.index(raw_label)             
                premise = instance['input']
                premise = task_prefix + premise

                if getattr(args, 'multiple_choice_prompt', None) is not None:
                    # Question: "Which of the following is a humorous edit of this artist or movie name: 'star wars'?"
                    # A. stare wars
                    # B. stariwars
                    # C. ...
                    # D. ...
                    # Answer:
                    hypotheses = options_sym
                    # premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
                    premise = f"{args.multiple_choice_prompt} Question: {premise}\n"
                    for idx in range(num_options):
                        premise += f"{options_sym[idx]}. {options_text[idx]}\n" 
                    premise += "Answer:"
                else:
                    hypotheses = options_text
                    premise = premise + uncond_premise
                example = [{
                    'label': label,
                    'premise': premise,
                    'uncond_premise': uncond_premise,
                }]
                for idx in range(num_options):
                    example[0][f'hypothesis{idx}'] = hypotheses[idx]

                examples += example
    return examples

def anli_loader(path, args):
    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"
    examples = []
    options_text = ["entailment", "neutral", "contradiction"]
    options_sym = ["A", "B", "C"]
    num_options = 3

    for one_path in path:
        with open(one_path) as lines:
            for line in lines:
                line = json.loads(line)
                label = ['e', 'n', 'c'].index(line['label'])       
                premise = f"{line['context']} {line['hypothesis']}"
                
                if getattr(args, 'multiple_choice_prompt', None) is not None:
                    # Question: "Which of the following is a humorous edit of this artist or movie name: 'star wars'?"
                    # A. entailment
                    # B. neutral
                    # C. contradiction
                    # Answer:
                    hypotheses = options_sym
                    # premise = f"{args.multiple_choice_prompt} Question: {premise}\nA. {options_text[0]}\nB. {options_text[1]}\nC. {options_text[2]}\nD. {options_text[3]}\nE. {options_text[4]}\nAnswer:"
                    premise = f"{args.multiple_choice_prompt} Question: {premise}\n"
                    for idx in range(num_options):
                        premise += f"{options_sym[idx]}. {options_text[idx]}\n" 
                    premise += "Answer:"                
                else:
                    hypotheses = options_text
                    premise = premise + uncond_premise
                example = [{
                    'label': label,
                    'premise': premise,
                    'uncond_premise': uncond_premise,
                }]
                for idx in range(num_options):
                    example[0][f'hypothesis{idx}'] = hypotheses[idx]
                examples += example

    return examples

def generate_n_shot_demonstrations(n_shot_dataset):
    n_shot_demonstrations = ""
    for raw_instance in n_shot_dataset:
        presmise = raw_instance['premise']
        answer_index = raw_instance['label'].item()
        answer = raw_instance[f"hypothesis{answer_index}"]
        n_shot_instance = f"{presmise}{answer}\n\n"
        n_shot_demonstrations += n_shot_instance
    return n_shot_demonstrations

def create_n_shot_splits(raw_dataset, n_shot_dataset, args):
    n_shot_demonstrations = ""
    if args.n_shot > 0:
        # few-shot setting: sample from train split, dev split (COPA), or the only split (BB)
        if n_shot_dataset is raw_dataset: # BB tasks: sampling from the only split, and use the rest
            raw_dataset = raw_dataset.train_test_split(test_size=args.n_shot, seed=args.seed)
            raw_dataset, n_shot_dataset = raw_dataset["train"], raw_dataset["test"]
        else:
            n_shot_dataset = n_shot_dataset.shuffle(seed=args.seed).select(range(args.n_shot))
        n_shot_demonstrations = generate_n_shot_demonstrations(n_shot_dataset)

    if args.sample is not None and args.sample <= len(raw_dataset):
        # sample "sample" amount of data from raw_data
        raw_dataset = raw_dataset.shuffle(seed=args.seed).select(range(args.sample))
    
    if args.n_shot > 0:
        # append n_shot_demonstrations to each input.
        raw_dataset = raw_dataset.map(lambda x: {"premise": n_shot_demonstrations + x["premise"]}) 
    else: # zero-shot: no need to return n_shot_dataset
        n_shot_dataset = None
    return raw_dataset, n_shot_dataset, n_shot_demonstrations

def generate_n_shot_poe_demonstrations(n_shot_dataset, num_of_options):
    # designed for multiple_choice_prompt
    alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:num_of_options]
    last_option = alphabets[-1]
    null_string = f"[MASK]"
    n_shot_demonstrations = ""
    n_shot_poe_demonstrations = ""
    for raw_instance in n_shot_dataset:
        premise = raw_instance['premise']
        answer_index = raw_instance['label'].item()
        answer = raw_instance[f"hypothesis{answer_index}"]
        n_shot_instance = f"{premise}{answer}\n\n"
        n_shot_demonstrations += n_shot_instance

        # for mcp: randomly mask options to [MASK]
        poe_premise = premise 
        new_alphabets = alphabets.replace(alphabets[answer_index], "")
        num_of_mask_options = random.randint(1, num_of_options-1)
        mask_option_symbols = random.sample(new_alphabets, num_of_mask_options) # e.g., [B, C]
        for symbol in mask_option_symbols:
            option_start_index = poe_premise.rfind(f"{symbol}. ")
            if symbol == last_option:
                option_end_index = poe_premise.rfind(f"Answer:")
            else:
                option_end_index = poe_premise.rfind(f"{alphabets[alphabets.index(symbol) + 1]}. ") 
            option = poe_premise[option_start_index:option_end_index]
            poe_premise = poe_premise.replace(option, f"{symbol}. {null_string}\n")
            
        n_shot_poe_instance = f"{poe_premise}{answer}\n\n"        
        n_shot_poe_demonstrations += n_shot_poe_instance
    return n_shot_demonstrations, n_shot_poe_demonstrations

def vqa_loader(path, args):
    versionType = '' # this should be '' when using VQA v2.0 dataset
    taskType = 'MultipleChoice' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
    dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
    dataSubType = 'train2014'
    annFile = '%s/Annotations/%s%s_%s_annotations.json'%(path, versionType, dataType, dataSubType)
    quesFile = '%s/Questions/%s%s_%s_%s_questions.json'%(path, versionType, taskType, dataType, dataSubType)
    imgDir = '%s/Images/%s/%s' %(path, dataType, dataSubType)

    examples = []

    print('Loading annotations and questions...')
    train_anno = json.load(open(annFile, 'r'))
    train_ques = json.load(open(quesFile, 'r'))

    if args.calibration_prompt is not None:
        uncond_premise = args.calibration_prompt
    else:
        uncond_premise = " the answer is:"

    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        img_id = train_anno['annotations'][i]['image_id']
        # question_id = train_anno['annotations'][i]['question_id']
        image_path = os.path.join(imgDir, 'COCO_train2014_' + '%012d.jpg' % img_id)

        question = train_ques['questions'][i]['question']
        mc_ans = train_ques['questions'][i]['multiple_choices']
        label = mc_ans.index(ans)

        if getattr(args, 'multiple_choice_prompt', None) is not None:
            hypotheses = mc_ans
            # Question: How does a bishop move from one place to another?
            # 1. chess game
            # 2. church
            # 3. in a car
            # 4. queen
            # 5. cathedral
            # Answer:
            options = "\n".join([f"{i}. {ans}" for i, ans in enumerate(mc_ans)])
            premise = f"{args.multiple_choice_prompt} Question: {question}\n{options}\nAnswer:"
        else:
            hypotheses = mc_ans
            premise = question + uncond_premise

        example = [{
            'premise': premise, 
            'image_path': image_path, 
            'uncond_premise': uncond_premise,  
            'label': label
            }]
    
        for idx, ans in enumerate(mc_ans):
            example[0][f'hypothesis{idx}'] = ans

        if i==0:
            print(example)

        examples+=example
    
    return examples