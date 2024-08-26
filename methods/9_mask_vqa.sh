#!/bin/bash
seeds=(0 1 2 3 4)
model_family="BLIP2"  # "BLIP2" "InstructBLIP" "GIT" "PaliGemma" "Idefics2"
checkpoint="Salesforce/blip2-opt-2.7b" # "Salesforce/blip2-opt-2.7b" "Salesforce/blip2-flan-t5-xl" "google/paligemma-3b-ft-science-qa-448" "google/paligemma-3b-ft-vqav2-448" "google/paligemma-3b-ft-ai2d-448"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
datasets="ai2d" # vqa scienceqa ai2d
batch_size=2
sample=100
n_shot=0

multiple_choice_prompt="Select the most suitable option to answer the question."
calibration_prompt=" the answer is:"
process_of_elimination_prompt="Select the most suitable option to answer the question. Ignore [MASK] options."
scoring_method_for_process_of_elimination=("channel" "calibration" "language_modeling" "multiple_choice_prompt")
mask_strategy_for_process_of_elimination=("below_average" "lowest")

for seed in "${seeds[@]}"; do
    for scoring_method in "${scoring_method_for_process_of_elimination[@]}"; do
        for mask_strategy in "${mask_strategy_for_process_of_elimination[@]}"; do

        # process of elimination
        python process_of_elimination.py \
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