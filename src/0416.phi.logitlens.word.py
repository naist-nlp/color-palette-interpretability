#!/usr/bin/env python
import torch
import json
import argparse
import os
import logging
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import *
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

# Configure logging for detailed debug output.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_model(model_name, device, hf_token):
    # Login with the Hugging Face token
    login(hf_token)
    
    logger.debug("Loading model '%s' on device '%s'", model_name, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",  # 自動でデータ型を選択
        _attn_implementation='eager'
    ).to(device).eval()
    
    logger.debug("Loading processor for model '%s'", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_crops=16
    )
    
    # Tokenizer に get_max_length がない場合、monkey patch を適用
    if not hasattr(processor.tokenizer, 'get_max_length'):
        logger.debug("Tokenizer does not have attribute 'get_max_length'. Applying monkey patch.")
        processor.tokenizer.get_max_length = lambda: getattr(processor.tokenizer, 'model_max_length', None)
    else:
        logger.debug("Tokenizer already has a 'get_max_length' attribute.")
    
    logger.debug("Tokenizer type: %s", type(processor.tokenizer))
    logger.debug("Tokenizer model_max_length: %s", getattr(processor.tokenizer, 'model_max_length', 'Not Available'))
    
    return model, processor

def load_prompts(prompt_file):
    logger.debug("Loading prompts from file: %s", prompt_file)
    with open(prompt_file, 'r') as f:
        return json.load(f)

def process_images(input_data, image_dir, output_file, model, processor, prompt, device):
    output_data = []

    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])
        logger.debug("Processing image: %s", img_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
            logger.debug("Image %s loaded and converted to RGB.", img_path)
            
            # ここで、JSON の "task_number" を利用してプロンプトテンプレート中の "{task_number}" を置換
            final_prompt = prompt.replace("{task_number}", item["task_number"])
            logger.debug("Final prompt after replacement: %s", final_prompt)
            
            # チャットテンプレート用メッセージ。ここでは final_prompt に加え、<|image_1|> のプレースホルダーを使用。
            messages = [
                {"role": "user", "content": f"{final_prompt}\n<|image_1|>\n"},
                {"role": "assistant", "content": ""}
            ]
            logger.debug("Generated messages: %s", messages)
            
            # チャットテンプレートを適用してプロンプトテキストを生成
            prompt_text = processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            logger.debug("Prompt text generated: %s", prompt_text)
            
            # プロンプトテキストと画像からモデル入力を作成
            inputs = processor(
                prompt_text, 
                [image], 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(device)
            logger.debug("Processor inputs keys: %s", inputs.keys())
            
            output_norm = model.model.norm
            unembedding_matrix = model.lm_head.weight.T  # (vocab_size, model_dim) -> (model_dim, vocab_size)

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
                  n_pls_1_input_text,
                  [image],
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
              generated_text = n_answer_token + n_pls_1_answer_token
              print(f"Answer: {generated_text}")
              layer_outputs.append({
                  "layer": layer+1,
                  "n_answer_token": generated_text,
                  "n_answer_prob": n_answer_prob,
                  "n_pls_1_answer_token": n_pls_1_answer_token,
                  "n_pls_1_answer_prob": n_pls_1_answer_prob
              })
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error("File error processing image %s: %s", img_path, e)
            generated_text = "Image not found"
        except Exception as e:
            logger.exception("Error processing image %s", img_path)
            generated_text = "Error generating response"

        item['model_out'] = generated_text
        # 出力結果を元データに追加
        item['layer_outputs'] = layer_outputs
        logitlens_dir = '/cl/home2/shintaro/color-palette-interpretability/logitlens.word.phi'
        os.makedirs(logitlens_dir, exist_ok=True)
        logitlens_save_file = os.path.join(logitlens_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_logitlens.png")
        visualize_logitlens_from_item(item, logitlens_save_file)
        item['logitlens'] = logitlens_save_file
        output_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logger.info("Processed images. Output saved to %s", output_file)

if __name__ == "__main__":
    seed_everything(42)
    
    parser = argparse.ArgumentParser(
        description="Run phi model on Ishihara images with debugging enabled."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--huggingface_token", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    # --prompt_type を削除（doctor用の固定promptを使用）
    parser.add_argument("--prompt_file", type=str, required=True)

    args = parser.parse_args()

    logger.info("Starting phi image processing with debugging enabled.")
    
    # モデルとプロセッサの初期化
    model, processor = initialize_model(args.model_name, args.device, args.huggingface_token)
    
    # === past_key_values エラー回避のためのパッチを適用 ===
    original_prepare_inputs_for_generation = model.prepare_inputs_for_generation

    def patched_prepare_inputs_for_generation(input_ids, **model_kwargs):
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            past = model_kwargs["past_key_values"]
            if not hasattr(past, "get_max_length"):
                past.get_max_length = lambda: getattr(processor.tokenizer, "model_max_length", 131072)
                logger.debug("Patched past_key_values with get_max_length: %s", past.get_max_length())
        return original_prepare_inputs_for_generation(input_ids, **model_kwargs)

    model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    logger.debug("Successfully patched model.prepare_inputs_for_generation")
    # === パッチ適用ここまで ===
    
    prompts = load_prompts(args.prompt_file)
    prompt = prompts["doctor_prompt_template"]
    
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)
    
    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt, args.device)
