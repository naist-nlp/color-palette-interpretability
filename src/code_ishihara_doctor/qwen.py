import torch
import json
import argparse
import os
import traceback
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import random
import numpy as np

def seed_everything(seed: int) -> None:
    """全ての乱数生成を固定して再現性を担保する"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def qwen_generate_response(model, processor, user_text, image, device="cuda", max_length=2048):
    try:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""}
        ]

        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]

        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text

    except Exception as e:
        print("[ERROR] in qwen_generate_response:", e)
        traceback.print_exc()
        return "Error generating response"

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return json.load(f)

def process_images(input_data, image_dir, output_file, model, processor, prompt, device):
    output_data = []

    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])
        try:
            image = Image.open(img_path).convert("RGB")
            # ここで、JSON の "task_number" を利用して prompt テンプレート内の "{task_number}" を実際の値に置換します。
            final_prompt = prompt.replace("{task_number}", item["task_number"])
            # qwen_generate_response に置換後のプロンプト(final_prompt)を渡します。
            answer = qwen_generate_response(model, processor, final_prompt, image, device)
        except (FileNotFoundError, UnidentifiedImageError):
            answer = "Image not found"
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            answer = "Error generating response"

        item['model_out'] = answer
        output_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    seed_everything(42)
    
    parser = argparse.ArgumentParser(description="Run Qwen2.5 VL on Ishihara images with prompts.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--huggingface_token", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    # --prompt_type を削除（doctor 用の固定プロンプトを使用）
    parser.add_argument("--prompt_file", type=str, required=True)

    args = parser.parse_args()

    login(args.huggingface_token)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        token=args.huggingface_token
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        token=args.huggingface_token
    )

    prompts = load_prompts(args.prompt_file)
    prompt = prompts["doctor_prompt_template"]

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt, args.device)
