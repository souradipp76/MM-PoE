: '
proxy_prefix="https://ghproxy.com/"

# anli
mkdir data/anli/
cd data/anli/
wget https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip
unzip anli_v1.0.zip

cp anli_v1.0/R1/dev.jsonl R1_dev.jsonl
cp anli_v1.0/R2/dev.jsonl R2_dev.jsonl
cp anli_v1.0/R3/dev.jsonl R3_dev.jsonl

cp anli_v1.0/R1/train.jsonl R1_train.jsonl
cp anli_v1.0/R2/train.jsonl R2_train.jsonl
cp anli_v1.0/R3/train.jsonl R3_train.jsonl

# big-bench 
mkdir ../big_bench/
cd ../big_bench/

repo_prefix="https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/"

single_tasks=("disambiguation_qa" "code_line_description" "reasoning_about_colored_objects" "crass_ai" "evaluating_information_essentiality" "identify_math_theorems" "identify_odd_metaphor" "logical_args" "riddle_sense")
conceptual_combinations_subtasks=("contradictions" "emergent_properties" "fanciful_fictional_combinations" "homonyms" "invented_words" "surprising_uncommon_combinations")
strange_stories_subtasks=("boolean" "multiple_choice")
symbol_interpretation_subtasks=("adversarial" "emoji_agnostic" "name_agnostic" "plain" "tricky")
logical_deduction_subtasks=("three_objects" "five_objects" "seven_objects")

for task in "${single_tasks[@]}"
do
    wget -O "${task}.json" "${proxy_prefix}${repo_prefix}${task}/task.json"
done

for subtask in "${conceptual_combinations_subtasks[@]}"
do
    wget -O "conceptual_combinations_${subtask}.json" "${proxy_prefix}${repo_prefix}conceptual_combinations/${subtask}/task.json"
done

for subtask in "${strange_stories_subtasks[@]}"
do
    wget -O "strange_stories_${subtask}.json" "${proxy_prefix}${repo_prefix}strange_stories/${subtask}/task.json"
    
done

for subtask in "${symbol_interpretation_subtasks[@]}"
do
    wget -O "symbol_interpretation_${subtask}.json" "${proxy_prefix}${repo_prefix}symbol_interpretation/${subtask}/task.json"
done

for subtask in "${logical_deduction_subtasks[@]}"
do
    wget -O "logical_deduction_${subtask}.json" "${proxy_prefix}${repo_prefix}logical_deduction/${subtask}/task.json"
done

# cqa
mkdir ../cqa/
cd ../cqa/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -O dev.jsonl
wget https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl -O test_no_answers.jsonl  
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl -O train.jsonl

# siqa
mkdir ../siqa/
cd ../siqa/
wget https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip
unzip socialiqa-train-dev.zip
cp socialiqa-train-dev/dev* .
cp socialiqa-train-dev/train* .
'

# vqa
# mkdir data/
cd data/

mkdir vqa
cd vqa
mkdir Annotations
cd Annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip
unzip Annotations_Train_mscoco.zip
rm Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip
unzip Annotations_Val_mscoco.zip
rm Annotations_Val_mscoco.zip
mkdir ../Questions
cd ../Questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip
unzip Questions_Train_mscoco.zip
rm Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip
unzip Questions_Val_mscoco.zip
rm Questions_Val_mscoco.zip
mkdir ../Images
cd ../Images
mkdir mscoco
cd mscoco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
cd ../../..

# scienceqa
mkdir scienceqa
cd scienceqa
echo "Downloading scienceqa dataset..."
gdown --folder https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw
cd ScienceQA_DATA
unzip train.zip
rm train.zip
unzip val.zip
rm val.zip
unzip test.zip
rm test.zip
cd ../..

# ai2d
mkdir ai2d
cd ai2d
echo "Downloading ai2d dataset..."
wget http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip
unzip ai2d-all.zip
rm ai2d-all.zip
cd ../..
echo "Done."