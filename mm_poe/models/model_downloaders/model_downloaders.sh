# you may specify model families and checkpoints here
model_families=("GIT")
checkpoints=("microsoft/git-base-vqav2")

for model_family in "${model_families[@]}"
do
    for checkpoint in "${checkpoints[@]}"
    do
        python models/model_downloaders/model_downloaders.py \
            --model_family ${model_family} \
            --checkpoint ${checkpoint} \
            --output_dir "./models"
            # --download_all_checkpoints  
    done
done