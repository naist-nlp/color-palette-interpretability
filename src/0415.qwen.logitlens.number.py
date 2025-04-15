import torch
import json
import argparse
import os
import traceback
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils import *
seed_everything(42)

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        return json.load(f)


def process_images(input_data, image_dir, output_file, model, processor, prompt, device, prompt_type):
    output_data = []

    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])

        # try:
        image = Image.open(img_path).convert("RGB")
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
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
            return_tensors="pt"
        ).to(device)

        output_norm = model.model.norm
        unembedding_matrix = model.model.embed_tokens.weight.T

        output_saving = []
        def forward_hook(module, inputs, output):
          if isinstance(output, tuple):
            output = output[0]
          output_saving.append(output)

        hook_handles = []
        for k, layer in enumerate(model.model.layers):
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
          print(f"Layer {layer+1} / {len(output_saving)}")
          n_answer_idx = inputs["input_ids"].shape[1] - 1
          n_answer_hidden = output[0, n_answer_idx]
          n_answer_hidden = output_norm(n_answer_hidden)
          n_answer_logits = n_answer_hidden @ unembedding_matrix
          n_answer_logprob = torch.softmax(n_answer_logits, dim=-1)
          n_answer_prob = torch.max(n_answer_logprob).item()
          n_answer_token = processor.tokenizer.decode(n_answer_logits.argmax().item())
          print(f"Answer token: {n_answer_token}, prob: {n_answer_prob:.4f}")

          n_pls_1_input_text = prompt_text + n_answer_token
          n_pls_1_inputs = processor(
              text=n_pls_1_input_text,
              images=[image],
              return_tensors="pt",
              truncation=True,
          ).to(device)
          with torch.no_grad():
            n_pls_1_answer_logits = model(**n_pls_1_inputs).logits
          n_pls_1_answer_idx = n_pls_1_inputs["input_ids"].shape[1] - 1
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
        # except (FileNotFoundError, UnidentifiedImageError):
        #     answer = "Image not found"
        # except Exception as e:
        #     print(f"Error processing image {img_path}: {e}")
        #     answer = "Error generating response"

        item['model_out'] = answer
        item['layer_outputs'] = layer_outputs
        logitlens_dir = f'/cl/home2/shintaro/color-palette-interpretability/logitlens.number.qwen'
        os.makedirs(logitlens_dir, exist_ok=True)
        logitlens_save_file = os.path.join(logitlens_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_logitlens.{prompt_type}.png")
        visualize_logitlens_from_item(item, logitlens_save_file)
        item['logitlens'] = logitlens_save_file
        output_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)


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

    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt, args.device, args.prompt_type)
