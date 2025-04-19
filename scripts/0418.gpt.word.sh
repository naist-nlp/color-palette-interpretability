#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=doctor.gpt
#SBATCH -o logs/slurm-%x-%j.log

set -eau

project_root=$(pwd)
source $project_root/.env
source $project_root/.venv/bin/activate
cd $project_root

INPUT_FILE=${project_root}/data/gold_answer/ishihara_doctor.json
OUTPUT_DIR=$project_root/ishihara_out/doctor.gpt
PROMPT_FILE=${project_root}/data/prompts/ishihara_doctor_prompt_forgpt.json
IMAGE_DIR=${project_root}/data/img_ishihara_plates
mkdir -p $OUTPUT_DIR

OUTPUT_FILE="${OUTPUT_DIR}/gpt.json"

python $project_root/src/0417.gpt.word.py \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE \
    --image_dir $IMAGE_DIR \
    --prompt_file $PROMPT_FILE

echo "Processed Ishihara test with gpt, output saved to ${OUTPUT_FILE}"

wait
