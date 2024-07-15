# you may specify model families and checkpoints here
model_families=("VILT")
checkpoints=("dandelin/vilt-b32-mlm")

for model_family in "${model_families[@]}"
do
    for checkpoint in "${checkpoints[@]}"
    do
        python models/model_downloaders/model_downloaders.py \
            --model_family ${model_family} \
            --checkpoint ${checkpoint} \
            --output_dir "/content/models"
            # --download_all_checkpoints  
    done
done