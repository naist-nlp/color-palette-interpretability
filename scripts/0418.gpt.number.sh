#!/bin/bash
#SBATCH -p lang_long
#SBATCH -c 4
#SBATCH -t 100:00:00
#SBATCH --account=lang
#SBATCH --job-name=gpt
#SBATCH -o logs/slurm-%x-%j.log

set -eau

project_root=$(pwd)
source $project_root/.env
source $project_root/.venv/bin/activate
cd $project_root

INPUT_FILE=${project_root}/data/gold_answer/ishihara.json
OUTPUT_DIR=$project_root/ishihara_out/gpt
PROMPT_FILE="${project_root}/data/prompts/ishihara_prompt_forgpt.json"
IMAGE_DIR=${project_root}/data/img_ishihara_plates
mkdir -p $OUTPUT_DIR

# Process each model and color vision type
for PROMPT_TYPE in protanopia deuteranopia tritanopia normal; do
    OUTPUT_FILE="${OUTPUT_DIR}/gpt_${PROMPT_TYPE}.json"

    # Conditionally add --quantize_type only if it's defined for the model
    python $project_root/src/0417.gpt.number.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --image_dir "$IMAGE_DIR" \
        --prompt_type "$PROMPT_TYPE" \
        --prompt_file "$PROMPT_FILE"
    echo "Processed Ishihara test with gpt model and ${PROMPT_TYPE} prompt, output saved to ${OUTPUT_FILE}"
done

wait
