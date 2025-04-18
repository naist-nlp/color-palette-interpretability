#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=is-nlp
#SBATCH --job-name=llava.logitlens.word
#SBATCH -o logs/slurm-%x-%j.log

set -eau

project_root=$(pwd)
source $project_root/.env
source $project_root/.venv/bin/activate
cd $project_root

INPUT_FILE=${project_root}/data/gold_answer/ishihara_doctor.json
OUTPUT_DIR=$project_root/ishihara_out/doctor.llava
HUGGINGFACE_TOKEN="$HF_TOKEN"
PROMPT_FILE="${project_root}/data/prompts/ishihara_doctor_prompt.json"
IMAGE_DIR=${project_root}/data/img_ishihara_plates
mkdir -p $OUTPUT_DIR

declare -A MODELS=(
    ["0416.llava.logitlens.word"]="llava-hf/llava-v1.6-vicuna-13b-hf"
)

declare -A QUANTIZE_TYPES=(
    ["0416.llava.logitlens.word"]="4bit"
)

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_KEY]}"
    QUANTIZE_TYPE="${QUANTIZE_TYPES[$MODEL_KEY]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_KEY}.json"

    echo "Running ${MODEL_KEY}..."

    CMD="python $project_root/src/${MODEL_KEY}.py \
        --input_file \"$INPUT_FILE\" \
        --output_file \"$OUTPUT_FILE\" \
        --model_name \"$MODEL_NAME\" \
        --huggingface_token \"$HUGGINGFACE_TOKEN\" \
        --image_dir \"$IMAGE_DIR\" \
        --prompt_file \"$PROMPT_FILE\""

    if [ -n "$QUANTIZE_TYPE" ]; then
        CMD+=" --quantize_type \"$QUANTIZE_TYPE\""
    fi

    eval $CMD

    echo "Processed Ishihara test with ${MODEL_KEY} model, output saved to ${OUTPUT_FILE}"

done

wait
