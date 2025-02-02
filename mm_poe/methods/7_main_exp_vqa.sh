#!/bin/bash
seeds=(0 1 2 3 4)
model_family="GIT" # "BLIP2-OPT" "BLIP2-T5" "InstructBLIP" "GIT" "PaliGemma" "Idefics2" 
checkpoints=("microsoft/git-base-vqav2")
loading_precision="FP32" # FP32 FP16 BF16(for 7b models) INT8
datasets="vqa scienceqa ai2d" # vqa scienceqa ai2d custom_dataset
batch_size=2
sample=100
n_shot=0

multiple_choice_prompt=""
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."

for seed in "${seeds[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
    # vision language modeling and average vision language modeling
    python vision_language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \
        
    # channel
    python vision_language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --do_channel \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \

    # multiple choice prompt, using the same script as language modeling
    python vision_language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --multiple_choice_prompt "$multiple_choice_prompt" \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \
    
    # calibration, i.e., PMI and PMI_DC.
    python vision_language_modeling.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --loading_precision ${loading_precision} \
        --calibration_prompt "${calibration_prompt}" \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \

    # process of elimination
    python process_of_elimination_vqa.py \
        --seed ${seed} \
        --model_family ${model_family} \
        --checkpoint ${checkpoint} \
        --loading_precision ${loading_precision} \
        --datasets "$datasets" \
        --batch_size  ${batch_size} \
        --multiple_choice_prompt "$multiple_choice_prompt" \
        --process_of_elimination_prompt "${process_of_elimination_prompt}" \
        --scoring_method_for_process_of_elimination "multiple_choice_prompt" \
        --mask_strategy_for_process_of_elimination "below_average" \
        --n_shot ${n_shot} \
        --sample ${sample} \
        # --push_data_to_hub \
    done
done