# you may specify model families and checkpoints here
model_families=("InstructBLIP")
checkpoints=("Salesforce/instructblip-vicuna-7b")

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