import torch
import random
import json
import argparse
import numpy as np
import os
import ast
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
from huggingface_hub import login

def seed_everything(seed: int) -> None:
    """
    Seed everything
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def initialize_model(model_name, attn_impl, torch_dtype, device, hf_token):
    """
    mPLUG-Owl3 モデルを初期化する関数
    """
    login(hf_token)

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        token=hf_token
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )

    # モデル独自の processor を初期化
    processor = model.init_processor(tokenizer)

    return model, processor, tokenizer

def load_prompts(prompt_file):
    """
    プロンプト設定が記載された JSON ファイルを読み込み、
    種類ごとに辞書で返す
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

def process_images(input_data, image_dir, output_file, model, processor, tokenizer, prompt_template, prompt_type, device):
    """
    few-shot (2画像) のプロンプトでモデル推論し、結果を JSON に出力
    """
    for item in input_data:
        # few-shot 用画像と本番画像のパスを取得
        few_shot_img_path = os.path.join(image_dir, item['few-shot-image_path'])
        question_img_path = os.path.join(image_dir, item['image_path'])

        # few-shot で使用する数値 (prompt_type ごとに取り出す)
        few_shot_number = item['few-shot-number'][prompt_type]

        try:
            # 画像読み込み
            few_shot_image = Image.open(few_shot_img_path).convert("RGB")
            question_image = Image.open(question_img_path).convert("RGB")

            # プロンプト文字列をフォーマット
            prompt_text = prompt_template.format(few_shot_number=few_shot_number)

            # 今回は2枚の画像を使うので、 <|image|> を2回入れたテキストにする
            messages = [
                {
                    "role": "user",
                    "content": f"<|image|>\n{prompt_text}\n<|image|>"
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ]

            # 画像はリストで指定 [few_shot_image, question_image]
            inputs = processor(messages, images=[few_shot_image, question_image], videos=None)

            # GPU に乗せる
            try:
                inputs = inputs.to(device)
            except Exception:
                # 一部のキーだけ to(device) する
                for key, value in inputs.items():
                    if hasattr(value, 'to'):
                        inputs[key] = value.to(device)

            # 生成パラメータを付与
            inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens': 1000,
                'decode_text': True,
            })

            # 推論実行
            g = model.generate(**inputs)

            # 単純に文字列化して answer に格納
            answer = g if isinstance(g, str) else str(g)

        except Exception as e:
            # 画像読み込みや推論で何かあればエラー文を保存
            answer = f"Error processing images: {e}"

        # もし生成結果がリスト表現っぽい場合、先頭要素だけを文字列化して返す
        answer_cleaned = answer.strip()
        try:
            parsed = ast.literal_eval(answer_cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                answer_cleaned = str(parsed[0]).strip()
        except Exception:
            pass

        # 結果を item に格納
        item["model_out"] = answer_cleaned

    # 最終的に結果を JSON 書き出し
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 実行時に seed を 42 に固定
    seed_everything(42)
    
    parser = argparse.ArgumentParser(
        description="mPLUG-Owl3 Chat with Images Generation (Few-shot 対応版)"
    )
    parser.add_argument("--model_name", type=str, required=True, help="mPLUG-Owl3 のモデルパスまたは名称")
    parser.add_argument("--attn_impl", type=str, choices=['sdpa', 'flash_attention_2'], default='sdpa', help="attention 実装")
    parser.add_argument("--torch_dtype", type=str, choices=['half', 'bfloat16'], default='half', help="Torch のデータ型")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="実行デバイス")
    parser.add_argument("--image_dir", type=str, required=True, help="画像が格納されたディレクトリ")
    parser.add_argument("--input_file", type=str, required=True, help="画像情報を含む入力JSONファイル")
    parser.add_argument("--output_file", type=str, required=True, help="生成結果を出力するJSONファイル")
    parser.add_argument("--prompt_type", type=str, required=True,
                        choices=["protanopia", "deuteranopia", "tritanopia", "normal"],
                        help="使用するプロンプトの種類")
    parser.add_argument("--prompt_file", type=str, required=True, help="プロンプト設定が記載されたJSONファイル")
    parser.add_argument("--hf_token", type=str, required=True, help="Huggingface Access Token")

    args = parser.parse_args()

    # torch_dtype の設定
    dtype = torch.half if args.torch_dtype == "half" else torch.bfloat16
    device = args.device

    # モデル等の初期化
    model, processor, tokenizer = initialize_model(
        args.model_name,
        args.attn_impl,
        dtype,
        device,
        args.hf_token
    )

    # プロンプトの辞書読み込み
    prompts = load_prompts(args.prompt_file)
    # 選択された prompt_type のテンプレートを取得
    prompt_template = prompts[args.prompt_type]

    # 入力 JSON の読み込み
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # few-shot 用に2枚の画像を読み込みつつ推論実行
    process_images(
        input_data,
        args.image_dir,
        args.output_file,
        model,
        processor,
        tokenizer,
        prompt_template,
        args.prompt_type,
        device
    )
