#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 16
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=is-nlp
#SBATCH --job-name=llama.logitlens
#SBATCH -o logs/slurm-%x-%j.log
set -eau

project_root=$(pwd)
source $project_root/.env
source $project_root/.venv/bin/activate
cd $project_root

INPUT_FILE=${project_root}/data/gold_answer/ishihara.json
OUTPUT_DIR=$project_root/ishihara_out/llama
HUGGINGFACE_TOKEN="$HF_TOKEN"
PROMPT_FILE="${project_root}/data/prompts/ishihara_prompt.json"
IMAGE_DIR=${project_root}/data/img_ishihara_plates
mkdir -p $OUTPUT_DIR

declare -A MODELS=(
    ["0415.llama.logitlens.number"]="meta-llama/Llama-3.2-11B-Vision-Instruct"
)

declare -A QUANTIZE_TYPES=(
    ["0415.llama.logitlens.number"]="half"
)

# to yutaka
#上記にモデル名と量子化全部まとめて入れれば一気に５つのモデル動かせる．["llama"]の部分は .pyのPrefixになってる！


# Process each model and color vision type
for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_KEY]}"
    QUANTIZE_TYPE="${QUANTIZE_TYPES[$MODEL_KEY]}"
    
    for PROMPT_TYPE in protanopia deuteranopia tritanopia normal; do
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_KEY}_${PROMPT_TYPE}.json"
        
        # Conditionally add --quantize_type only if it's defined for the model
        if [ -n "$QUANTIZE_TYPE" ]; then
            python $project_root/src/${MODEL_KEY}.py \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --model_name "$MODEL_NAME" \
                --quantize_type "$QUANTIZE_TYPE" \
                --huggingface_token "$HUGGINGFACE_TOKEN" \
                --image_dir "$IMAGE_DIR" \
                --prompt_type "$PROMPT_TYPE" \
                --prompt_file "$PROMPT_FILE"
        else
            python $project_root/src/${MODEL_KEY}.py \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --model_name "$MODEL_NAME" \
                --huggingface_token "$HUGGINGFACE_TOKEN" \
                --image_dir "$IMAGE_DIR" \
                --prompt_type "$PROMPT_TYPE" \
                --prompt_file "$PROMPT_FILE"
        fi

        echo "Processed Ishihara test with ${MODEL_KEY} model and ${PROMPT_TYPE} prompt, output saved to ${OUTPUT_FILE}"
    done
done

wait
