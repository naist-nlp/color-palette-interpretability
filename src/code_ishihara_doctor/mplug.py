import torch
import json
import argparse
import os
import ast
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
from huggingface_hub import login

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

def initialize_model(model_name, attn_impl, torch_dtype, device, huggingface_token):
    print("DEBUG: Logging in with HF token")
    login(huggingface_token)

    print(f"DEBUG: Loading config for model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        token=huggingface_token
    ).to(device).eval()
    print("DEBUG: Model loaded and moved to device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=huggingface_token)
    print("DEBUG: Tokenizer loaded")
    
    # モデル独自の processor を初期化
    processor = model.init_processor(tokenizer)
    print("DEBUG: Processor initialized")
    
    return model, processor, tokenizer

def load_prompts(prompt_file):
    print("DEBUG: Loading prompt file:", prompt_file)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    print("DEBUG: Prompts loaded:", list(prompts.keys()))
    return prompts

def process_images(input_data, image_dir, output_file, model, processor, tokenizer, prompt, device):
    # 各入力アイテムに生成結果（model_out）を追加して更新する
    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])
        print("\nDEBUG: Processing image:", img_path)
        try:
            print("DEBUG: Opening image")
            image = Image.open(img_path).convert("RGB")
            print("DEBUG: Image loaded successfully")

            # ここで prompt 内の "{task_number}" を実際の task_number で置換します
            final_prompt = prompt.replace("{task_number}", item["task_number"])

            # ユーザーメッセージに final_prompt を組み込む
            messages = [
                {"role": "user", "content": f"<|image|>\n{final_prompt}"},
                {"role": "assistant", "content": ""}
            ]
            print("DEBUG: Messages prepared:", messages)
            
            print("DEBUG: Calling processor(...)")
            inputs = processor(messages, images=[image], videos=None)
            print("DEBUG: Processor returned inputs. Keys and types:")
            for key, value in inputs.items():
                print(f"    Key: {key}, Type: {type(value)}")
            
            # Quick start を参考に、processor の戻り値に対して直接 .to(device) を呼び出す
            try:
                inputs = inputs.to(device)
                print("DEBUG: Converted inputs to device:", device)
            except Exception as to_error:
                print("DEBUG: inputs.to(device) の実行に失敗、各テンソルを手動で変換します。エラー:", to_error)
                for key, value in inputs.items():
                    if hasattr(value, 'to'):
                        inputs[key] = value.to(device)
            
            # 追加パラメータの更新（Quick start 同様）
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens': 50,
                'decode_text': True,
            })
            print("DEBUG: Inputs after update:", list(inputs.keys()))

            print("DEBUG: Calling model.generate()")
            try:
                g = model.generate(**inputs)
                print("DEBUG: model.generate() completed successfully")
            except Exception as gen_error:
                print("DEBUG: Error during model.generate():", gen_error)
                raise gen_error

            answer = g if isinstance(g, str) else str(g)
            print("DEBUG: Answer obtained:", answer[:100])
        except Exception as e:
            print(f"DEBUG: Error processing image {img_path}: {e}")
            answer = "Error processing image"

        # もし answer が文字列であり、リテラルリスト表現になっていれば先頭要素を抽出
        answer_cleaned = answer.strip()
        try:
            parsed = ast.literal_eval(answer_cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                answer_cleaned = str(parsed[0]).strip()
        except Exception:
            # 変換できなければそのまま使用
            pass

        # 入力アイテムに "model_out" キーとして生成結果を追加
        item["model_out"] = answer_cleaned

    print("DEBUG: Writing output to file:", output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)
    print("DEBUG: Output file written successfully.")

if __name__ == "__main__":
    seed_everything(42)
    
    parser = argparse.ArgumentParser(
        description="mPLUG-Owl3 Chat with Images Generation (出力は生成テキスト)"
    )
    parser.add_argument("--model_name", type=str, required=True, help="mPLUG-Owl3 のモデルパスまたは名称")
    parser.add_argument("--attn_impl", type=str, choices=['sdpa', 'flash_attention_2'], default='sdpa', help="attention 実装")
    parser.add_argument("--torch_dtype", type=str, choices=['half', 'bfloat16'], default='half', help="Torch のデータ型")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="実行デバイス")
    parser.add_argument("--image_dir", type=str, required=True, help="画像が格納されたディレクトリ")
    parser.add_argument("--input_file", type=str, required=True, help="画像情報を含む入力JSONファイル（各アイテムは 'image_path' キーを含む）")
    parser.add_argument("--output_file", type=str, required=True, help="生成結果を出力するJSONファイル")
    # --prompt_type を削除（doctor用の固定promptを使用）
    parser.add_argument("--prompt_file", type=str, required=True, help="プロンプト設定が記載されたJSONファイル")
    parser.add_argument("--huggingface_token", type=str, required=True, help="Huggingface Access Token")

    args = parser.parse_args()

    dtype = torch.half if args.torch_dtype == "half" else torch.bfloat16
    device = args.device

    print("DEBUG: Starting model initialization")
    model, processor, tokenizer = initialize_model(args.model_name, args.attn_impl, dtype, device, args.huggingface_token)

    print("DEBUG: Loading prompts")
    prompts = load_prompts(args.prompt_file)
    prompt = prompts["doctor_prompt_template"]
    print("DEBUG: Using prompt: doctor_prompt_template")

    print("DEBUG: Loading input data from", args.input_file)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print("DEBUG: Loaded input data. Number of items:", len(input_data))

    process_images(input_data, args.image_dir, args.output_file, model, processor, tokenizer, prompt, device)
