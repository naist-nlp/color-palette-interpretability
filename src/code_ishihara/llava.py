import json
import torch
import argparse
import os
from PIL import Image, UnidentifiedImageError
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig

def initialize_model(model_name, device, quantize_type, huggingface_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token, trust_remote_code=True)

    model_kwargs = {
        "token": huggingface_token,
        "trust_remote_code": True
    }

    if quantize_type == '4bit':
        model_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            "torch_dtype": torch.float16
        })
    elif quantize_type == '8bit':
        model_kwargs.update({
            "device_map": 'auto',
            "load_in_8bit": True,
            "use_cache": True
        })
    elif quantize_type == 'half':
        model_kwargs.update({
            "torch_dtype": torch.float16
        })

    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    if quantize_type not in ['4bit', '8bit']:
        model.to(device)
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(model_name, token=huggingface_token, trust_remote_code=True)

    return model, processor, tokenizer


def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return json.load(f)


def process_images(input_data, image_dir, output_file, model, processor, prompt, device):
    output_data = []

    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])

        try:
            image = Image.open(img_path).convert("RGB")

            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(images=image, text=input_text, return_tensors="pt").to(device)

            output = model.generate(**inputs, max_new_tokens=50)
            answer = processor.decode(output[0], skip_special_tokens=True)

        except (FileNotFoundError, UnidentifiedImageError):
            answer = "Image not found"
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            answer = "Error processing image"

        item['model_out'] = answer
        output_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLaVA-Next for VQA task on Ishihara test images.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--quantize_type", type=str, choices=['4bit', '8bit', 'half'], default="4bit")
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["protanopia", "deuteranopia", "tritanopia", "normal"])
    parser.add_argument("--prompt_file", type=str, required=True)

    args = parser.parse_args()

    model, processor, tokenizer = initialize_model(args.model_name, args.device, args.quantize_type, args.huggingface_token)

    prompts = load_prompts(args.prompt_file)
    prompt_text = prompts[args.prompt_type]

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt_text, args.device)
