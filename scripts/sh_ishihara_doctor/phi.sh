#!/bin/bash
#SBATCH --job-name=ishihara_color_palette
#SBATCH --output=/cl/home2/kazuki-ha/color/log/doc_phi_ishihara_%j.log
#SBATCH --partition=gpu_long
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a6000:1

source /home/is/kazuki-ha/yes/etc/profile.d/conda.sh
conda activate wow

cd /cl/home2/kazuki-ha/color/code_ishihara_doctor

export HF_TOKEN=""

INPUT_FILE="/cl/home2/kazuki-ha/color/ishihara_docter.json"
OUTPUT_DIR="/cl/home2/kazuki-ha/color/ishihara_out_doc/phi"
HUGGINGFACE_TOKEN="$HF_TOKEN"
PROMPT_FILE="/cl/home2/kazuki-ha/color/ishihara_prompt_doctor.json"
IMAGE_DIR="/cl/home2/kazuki-ha/color/ishihara_plates"

declare -A MODELS=(
    ["phi"]="microsoft/Phi-3-vision-128k-instruct"
)

declare -A QUANTIZE_TYPES=(
    ["phi"]=""  # phiはquantize無し
)

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[$MODEL_KEY]}"
    QUANTIZE_TYPE="${QUANTIZE_TYPES[$MODEL_KEY]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_KEY}.json"

    echo "Running ${MODEL_KEY}..."

    CMD="python ${MODEL_KEY}.py \
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
