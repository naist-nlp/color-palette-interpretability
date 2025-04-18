#!/usr/bin/env python
import torch
import json
import argparse
import os
import traceback
from PIL import Image, UnidentifiedImageError
from huggingface_hub import login
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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

def qwen_generate_response_multi_image(model, processor, user_text, image_list, device="cuda", max_length=2048):
    """
    Qwen2.5-VLで複数画像 + テキストを用いた推論を行うヘルパー関数。
    image_list は、[image1, image2, ...] のように PIL.Image.Image を格納したリスト。
    """
    try:
        # Qwen2.5-VL は "messages" 構造: "role": "user" / "assistant"、contentに[type: "image"/"text"]を並べる
        user_content = []
        for img in image_list:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""}
        ]

        # Chatテンプレートへ変換
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 画像と動画の情報に分解（今回は動画なしなので video_inputs は空になるはず）
        image_inputs, video_inputs = process_vision_info(messages)

        # Processorでtensor化
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
            outputs = model.generate(**inputs, max_new_tokens=128)

        # 入力プロンプト部分のトークンを取り除く
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
        ]

        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return generated_text

    except Exception as e:
        print("[ERROR] in qwen_generate_response_multi_image:", e)
        traceback.print_exc()
        return "Error generating response"


def load_prompts(prompt_file):
    """
    プロンプト設定 (JSON) を読み込み。
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_images_fewshot(input_data, image_dir, output_file, model, processor, prompt_template, prompt_type, device):
    """
    Few-shot 形式で 2枚の画像 (few-shot用 & 質問用) を同時入力し、回答を得る。
    """
    output_data = []

    # 事前に設定したいリサイズ幅・高さ（必要に応じて変更）
    resize_size = (512, 512)

    for item in input_data:
        # few-shot画像と質問画像のパス
        fs_img_path = os.path.join(image_dir, item["few-shot-image_path"])
        q_img_path = os.path.join(image_dir, item["image_path"])

        # few-shot 用の数字
        few_shot_number = item["few-shot-number"][prompt_type]

        # プロンプトに {few_shot_number} を差し込む
        user_prompt = prompt_template.format(few_shot_number=few_shot_number)

        # 画像を読み込む & リサイズ
        try:
            fs_image = Image.open(fs_img_path).convert("RGB")
            fs_image = fs_image.resize(resize_size)  # 画像サイズを縮小
            q_image = Image.open(q_img_path).convert("RGB")
            q_image = q_image.resize(resize_size)    # 画像サイズを縮小

            image_list = [fs_image, q_image]
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image(s) for item {item}: {e}")
            item["model_out"] = "Image not found"
            output_data.append(item)
            continue
        except Exception as e:
            print(f"Unknown error in reading images: {e}")
            item["model_out"] = "Error loading images"
            output_data.append(item)
            continue

        # 推論
        try:
            answer = qwen_generate_response_multi_image(
                model=model,
                processor=processor,
                user_text=user_prompt,
                image_list=image_list,
                device=device
            )
        except Exception as e:
            print(f"Error generating response for item {item}: {e}")
            answer = "Error generating response"

        item["model_out"] = answer
        output_data.append(item)

    # JSON で書き出し
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # ---- ここでシードを固定 ----
    seed_everything(42)

    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL in few-shot style with multiple images.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFaceモデルID (例: Qwen/Qwen2.5-VL-7B-Instruct).")
    parser.add_argument("--huggingface_token", type=str, required=True,
                        help="HuggingFaceのトークン.")
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="実行デバイス.")
    parser.add_argument("--image_dir", type=str, required=True, help="画像があるディレクトリ.")
    parser.add_argument("--input_file", type=str, required=True, help="入力JSONファイル.")
    parser.add_argument("--output_file", type=str, required=True, help="生成結果を出力するJSONファイル.")
    parser.add_argument("--prompt_type", type=str, required=True,
                        choices=["protanopia", "deuteranopia", "tritanopia", "normal"],
                        help="どのプロンプトを使うか.")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="プロンプトが記述されたJSONファイル.")

    args = parser.parse_args()

    # HuggingFaceにログイン
    login(args.huggingface_token)

    # Processorの追加設定例:
    #   - num_crops: multi-frameの場合は4推奨、single-frameなら16など
    #   - min_pixels, max_pixels で解像度制限し、GPU負荷を下げる
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        token=args.huggingface_token,
        num_crops=4,
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    # モデルのロード
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # メモリが十分であれば "auto" でOK。GPU1枚のみなら {"": args.device} でも良い。
        token=args.huggingface_token
    ).to(args.device)

    # プロンプトの読み込み
    prompts = load_prompts(args.prompt_file)
    prompt_template = prompts[args.prompt_type]  # 例: { "protanopia": "... {few_shot_number} ..." }

    # 入力データの読み込み
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # few-shot 推論実行
    process_images_fewshot(
        input_data=input_data,
        image_dir=args.image_dir,
        output_file=args.output_file,
        model=model,
        processor=processor,
        prompt_template=prompt_template,
        prompt_type=args.prompt_type,
        device=args.device
    )
