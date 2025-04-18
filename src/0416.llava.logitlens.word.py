import json
import torch
import argparse
import os
from PIL import Image, UnidentifiedImageError
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from utils import *
import random
import numpy as np

def seed_everything(seed: int) -> None:
    """全ての乱数生成のシードを固定する"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

            # 各タスクごとに、prompt テンプレート内の "{task_number}" を実際の task_number に置換
            final_prompt = prompt.replace("{task_number}", item["task_number"])

            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": final_prompt}]}
            ]

            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=image, text=input_text, return_tensors="pt").to(device)

            unembedding_matrix = model.language_model.lm_head.weight.T
            output_norm = model.language_model.model.norm

            output_saving = []
            def forward_hook(module, inputs, output):
              if isinstance(output, tuple):
                output = output[0]
              output_saving.append(output)

            hook_handles = []
            for k, layer in enumerate(model.language_model.model.layers):
              handle = layer.register_forward_hook(forward_hook)
              hook_handles.append(handle)

            with torch.no_grad():
              _ = model(**inputs)

            for handle in hook_handles:
              handle.remove()

            layer_outputs = []
            for layer, output in enumerate(output_saving):
              if layer < len(output_saving) // 2:
                continue
              n_answer_idx = inputs["input_ids"].shape[1] - 1
              n_answer_hidden = output[0, n_answer_idx]
              n_answer_hidden = output_norm(n_answer_hidden)
              n_answer_logits = n_answer_hidden @ unembedding_matrix
              n_answer_logprob = torch.softmax(n_answer_logits, dim=-1)
              n_answer_prob = torch.max(n_answer_logprob).item()
              n_answer_token = processor.tokenizer.decode(n_answer_logits.argmax().item())
              print(f"Answer token: {n_answer_token}, prob: {n_answer_prob:.4f}")

              n_pls_1_input_text = input_text + n_answer_token
              n_pls_1_inputs = processor(
                  text=n_pls_1_input_text,
                  images=[image],
                  return_tensors="pt",
                  truncation=True,
                  max_length=2048
              ).to(device)
              with torch.no_grad():
                n_pls_1_answer_logits = model(**n_pls_1_inputs).logits
              n_pls_1_answer_logprob = torch.softmax(n_pls_1_answer_logits, dim=-1)
              n_pls_1_answer_prob = torch.max(n_pls_1_answer_logprob).item()
              n_pls_1_answer_token = processor.tokenizer.decode(n_pls_1_answer_logits.argmax().item())
              print(f"Next token: {n_pls_1_answer_token}, prob: {n_pls_1_answer_prob:.4f}")
              answer = n_answer_token + n_pls_1_answer_token
              print(f"Answer: {answer}")
              layer_outputs.append({
                  "layer": layer+1,
                  "n_answer_token": answer,
                  "n_answer_prob": n_answer_prob,
                  "n_pls_1_answer_token": n_pls_1_answer_token,
                  "n_pls_1_answer_prob": n_pls_1_answer_prob
              })


        except (FileNotFoundError, UnidentifiedImageError):
            answer = "Image not found"
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            answer = "Error processing image"

        item['model_out'] = answer
        item['layer_outputs'] = layer_outputs
        logitlens_dir = '/cl/home2/shintaro/color-palette-interpretability/logitlens.word.llava'
        os.makedirs(logitlens_dir, exist_ok=True)
        logitlens_save_file = os.path.join(logitlens_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_logitlens.png")
        visualize_logitlens_from_item(item, logitlens_save_file)
        item['logitlens'] = logitlens_save_file
        output_data.append(item)



    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    seed_everything(42)
    
    parser = argparse.ArgumentParser(description="Run LLaVA-Next for VQA task on Ishihara test images.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--quantize_type", type=str, choices=['4bit', '8bit', 'half'], default="4bit")
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    # --prompt_type削除（固定promptを使用するため）
    parser.add_argument("--prompt_file", type=str, required=True)

    args = parser.parse_args()

    model, processor, tokenizer = initialize_model(args.model_name, args.device, args.quantize_type, args.huggingface_token)

    prompts = load_prompts(args.prompt_file)
    prompt_text = prompts["doctor_prompt_template"]

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt_text, args.device)
