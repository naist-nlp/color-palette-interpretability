import torch
import json
import argparse
import os
import traceback
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


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
            answer = qwen_generate_response(model, processor, prompt, image, device)

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
    parser = argparse.ArgumentParser(description="Run Qwen2.5 VL on Ishihara images with prompts.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--huggingface_token", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["protanopia", "deuteranopia", "tritanopia", "normal"])
    parser.add_argument("--prompt_file", type=str, required=True)

    args = parser.parse_args()

    # ココだけ修正
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
    prompt = prompts[args.prompt_type]

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt, args.device)
