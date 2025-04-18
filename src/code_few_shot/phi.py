#!/usr/bin/env python
import json
import torch
import argparse
import logging
import os
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoProcessor

# ---- ここから追加 ----
import random
import numpy as np

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
# ---- ここまで追加 ----

# ログ設定（DEBUGレベル）
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def initialize_model(model_name, device, hf_token):
    """
    Phi-3.5モデルおよびプロセッサの初期化関数。
    Hugging Faceへのログインを行い、指定デバイスへロードします。
    """
    if hf_token:
        login(hf_token)
        logger.debug("Logged in to Hugging Face with provided token.")
    
    logger.debug("Loading model '%s' on device '%s'", model_name, device)
    # flash_attention_2 が使えない環境であれば '_attn_implementation' を 'eager' にする
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='eager'  # 環境に応じて 'flash_attention_2' 等を指定
    ).eval()
    logger.debug("Model loaded successfully.")
    
    logger.debug("Loading processor for model '%s'", model_name)
    # single-frameの場合は num_crops=16、multi-frameの場合は num_crops=4 などを調整
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_crops=16
    )
    logger.debug("Processor loaded successfully.")
    
    # Tokenizer に get_max_length がない場合、monkey patch を適用
    if not hasattr(processor.tokenizer, 'get_max_length'):
        logger.debug("Tokenizer does not have attribute 'get_max_length'. Applying monkey patch.")
        processor.tokenizer.get_max_length = lambda: getattr(processor.tokenizer, 'model_max_length', None)
    else:
        logger.debug("Tokenizer already has 'get_max_length' attribute.")
    
    return model.to(device), processor

def load_image(image_path):
    """
    ローカルまたはURLから画像をロードしてRGBに変換する。
    """
    logger.debug("Loading image from: %s", image_path)
    try:
        # URLの場合
        if image_path.startswith("http://") or image_path.startswith("https://"):
            import requests
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")
        else:
            # ローカルファイルの場合
            image = Image.open(image_path).convert("RGB")
        logger.debug("Image loaded and converted to RGB successfully.")
        return image
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.error("Error loading image %s: %s", image_path, e)
        raise

def load_prompts(prompt_file):
    """
    prompt_file(JSON)からプロンプト設定を読み込み、返す。
    """
    logger.debug("Loading prompts from: %s", prompt_file)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_images(input_data, image_dir, output_file, model, processor, prompt_template, prompt_type, device):
    """
    few-shot (2枚の画像) 対応版の推論処理。
    input_data の各アイテムには:
      - "few-shot-image_path": few-shot例示画像
      - "image_path": 質問用画像
      - "few-shot-number": { "protanopia": ..., "deuteranopia": ..., ... } など
      - その他メタ情報
    などが含まれている想定。
    
    prompt_template に {few_shot_number} が含まれている場合は、
    item["few-shot-number"][prompt_type] を埋め込む形でプロンプト文字列を生成します。
    """
    output_data = []
    for item in input_data:
        # few-shot画像と本番画像を取得
        few_shot_path = os.path.join(image_dir, item["few-shot-image_path"])
        question_path = os.path.join(image_dir, item["image_path"])

        # few-shot用の数字を prompt_type から取得
        few_shot_number = item["few-shot-number"][prompt_type]
        
        # 実際に画像読み込み
        try:
            few_shot_image = load_image(few_shot_path)
            question_image = load_image(question_path)
        except Exception as e:
            logger.exception("Error loading images for item %s", item)
            item["model_out"] = f"Error loading images: {e}"
            output_data.append(item)
            continue
        
        # テンプレート文字列に few_shot_number を差し込み
        prompt_text_filled = prompt_template.format(few_shot_number=few_shot_number)

        # ユーザーメッセージ: 2つの画像プレースホルダとテキストをセット
        # 例: <|user|>\n<|image_1|>\n<|image_2|>\n{prompt_text} <|end|>\n<|assistant|>\n
        # assistant には空文字または最小限を入れておく
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n<|image_2|>\n{prompt_text_filled}<|end|>\n"
            },
            {
                "role": "assistant",
                "content": ""
            },
        ]
        
        # マルチターン用テンプレートを組み立て
        chat_prompt_text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.debug("Final chat prompt: %s", chat_prompt_text)

        # 画像をリストで与える: [few_shot_image, question_image]
        images = [few_shot_image, question_image]

        # Processor 入力を作成
        try:
            inputs = processor(
                chat_prompt_text,
                images,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(device)
        except Exception as e:
            logger.exception("Error creating processor inputs for item %s", item)
            item["model_out"] = f"Error creating inputs: {e}"
            output_data.append(item)
            continue
        
        # 生成パラメータ
        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
            # eos_token_id を設定する場合は下記のように:
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        # 推論実行
        try:
            with torch.no_grad():
                generate_ids = model.generate(**inputs, use_cache=False, **generation_args)
            
            # 入力プロンプト部分のトークンを削除
            if "input_ids" in inputs:
                input_length = inputs["input_ids"].shape[1]
                generate_ids = generate_ids[:, input_length:]
            
            # テキストへデコード
            generated_text = processor.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            logger.debug("Decoded model response: %s", generated_text)
        
        except Exception as e:
            logger.exception("Error during model generation for item %s", item)
            generated_text = f"Error during generation: {e}"
        
        # JSON出力用に生成結果を格納
        item["model_out"] = generated_text
        output_data.append(item)
    
    # 最終的に結果を JSON 書き出し
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    logger.info("Processed images. Output saved to %s", output_file)

if __name__ == "__main__":
    # ---- ここでシードを固定 ----
    seed_everything(42)

    parser = argparse.ArgumentParser(
        description="Run Phi-3.5 vision model (few-shot) on multiple images for research purposes."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3.5-vision-instruct",
        help="HuggingFaceのモデルID。"
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        required=True,
        help="HuggingFaceの認証トークン。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="モデル実行に用いるデバイス。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="入力JSONファイルのパス（画像情報が含まれる）。"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="出力JSONファイルのパス。"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["protanopia", "deuteranopia", "tritanopia", "normal"],
        help="使用するプロンプトタイプ。"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="プロンプトを記述したJSONファイルのパス。"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="画像が保存されているディレクトリのパス。"
    )
    
    args = parser.parse_args()
    
    logger.info("Initializing Phi-3.5 model and processor.")
    model, processor = initialize_model(args.model_name, args.device, args.huggingface_token)
    
    logger.info("Loading prompts from %s", args.prompt_file)
    prompts = load_prompts(args.prompt_file)
    prompt_template = prompts[args.prompt_type]  # 例: prompts["protanopia"]
    
    logger.info("Loading input data from %s", args.input_file)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    logger.info("Processing images with few-shot context...")
    process_images(
        input_data=input_data,
        image_dir=args.image_dir,
        output_file=args.output_file,
        model=model,
        processor=processor,
        prompt_template=prompt_template,
        prompt_type=args.prompt_type,
        device=args.device
    )
