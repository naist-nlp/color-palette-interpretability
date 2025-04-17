#!/usr/bin/env python
import torch
import json
import argparse
import os
import logging
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor

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
            
            # 生成パラメータ（max_new_tokens や temperature など）
            generation_args = {
                "max_new_tokens": 100,
                "temperature": 0.0,
                "do_sample": False,
            }
            
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    **generation_args
                )
            logger.debug("Generated IDs shape: %s", generate_ids.shape)
            
            generated_text = processor.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            logger.debug("Generated text: %s", generated_text)
        
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error("File error processing image %s: %s", img_path, e)
            generated_text = "Image not found"
        except Exception as e:
            logger.exception("Error processing image %s", img_path)
            generated_text = "Error generating response"
        
        item['model_out'] = generated_text
        output_data.append(item)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
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
