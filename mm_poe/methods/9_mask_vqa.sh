#!/bin/bash
seeds=(0 1 2 3 4)
model_family="GIT"  # "BLIP2-OPT" "BLIP2-T5" "InstructBLIP" "GIT" "PaliGemma" "Idefics2"
checkpoint="microsoft/git-base-vqav2"
loading_precision="FP32" # FP32 FP16 BF16(for 7b models) INT8
datasets="vqa scienceqa ai2d" # vqa scienceqa ai2d custom_dataset
batch_size=2
sample=100
n_shot=0

multiple_choice_prompt=""
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
scoring_method_for_process_of_elimination=("channel" "calibration" "language_modeling" "multiple_choice_prompt")
mask_strategy_for_process_of_elimination=("below_average" "lowest")

for seed in "${seeds[@]}"; do
    for scoring_method in "${scoring_method_for_process_of_elimination[@]}"; do
        for mask_strategy in "${mask_strategy_for_process_of_elimination[@]}"; do

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
            --scoring_method_for_process_of_elimination "$scoring_method" \
            --mask_strategy_for_process_of_elimination "$mask_strategy" \
            --n_shot ${n_shot} \
            --sample ${sample} \
            # --push_data_to_hub 
        
        done
    done
done