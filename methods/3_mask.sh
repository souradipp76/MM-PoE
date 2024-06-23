#!/bin/bash
seeds=(0 1 2 3 4)
model_family="FLAN-T5"  # "OPT-IML" "FLAN-T5" "Pythia" 
checkpoint="google/flan-t5-xl" # "EleutherAI/pythia-2.8b" "facebook/opt-iml-1.3b" "facebook/opt-iml-max-1.3b" "google/flan-t5-large" "google/flan-t5-xl"
loading_precision="FP16" # FP32 FP16 BF16(for 7b models) INT8
datasets="anli cqa siqa logical_deduction_five_objects disambiguation_qa conceptual_combinations strange_stories symbol_interpretation"
batch_size=16
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