#!/bin/bash
#SBATCH --job-name=ishihara_color_palette
#SBATCH --output=/cl/home2/kazuki-ha/color/log/qwen_ishihara_%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a6000:1

source /home/is/kazuki-ha/yes/etc/profile.d/conda.sh
conda activate wow
cd /cl/home2/kazuki-ha/color/code_ishihara
export HF_TOKEN="your"



INPUT_FILE="/cl/home2/kazuki-ha/color/ishihara.json"
OUTPUT_DIR="/cl/home2/kazuki-ha/color/ishihara_out/qwen"
HUGGINGFACE_TOKEN="$HF_TOKEN" 
PROMPT_FILE="/cl/home2/kazuki-ha/color/ishihara_prompt.json"
IMAGE_DIR="/cl/home2/kazuki-ha/color/ishihara_plates"

declare -A MODELS=(
    ["qwen"]="Qwen/Qwen2.5-VL-7B-Instruct"
)

declare -A QUANTIZE_TYPES=(
    ["qwen"]="half"
)

# Process each model and color vision type
for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_KEY]}"
    QUANTIZE_TYPE="${QUANTIZE_TYPES[$MODEL_KEY]}"
    
    for PROMPT_TYPE in protanopia deuteranopia tritanopia normal; do
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_KEY}_${PROMPT_TYPE}.json"
        
        # Conditionally add --quantize_type only if it's defined for the model
        if [ -n "$QUANTIZE_TYPE" ]; then
            python ${MODEL_KEY}.py \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --model_name "$MODEL_NAME" \
                --huggingface_token "$HUGGINGFACE_TOKEN" \
                --image_dir "$IMAGE_DIR" \
                --prompt_type "$PROMPT_TYPE" \
                --prompt_file "$PROMPT_FILE"
        else
            python ${MODEL_KEY}.py \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --huggingface_token "$HUGGINGFACE_TOKEN" \
                --image_dir "$IMAGE_DIR" \
                --prompt_type "$PROMPT_TYPE" \
                --prompt_file "$PROMPT_FILE"
        fi

        echo "Processed Ishihara test with ${MODEL_KEY} model and ${PROMPT_TYPE} prompt, output saved to ${OUTPUT_FILE}"
    done
done

wait
