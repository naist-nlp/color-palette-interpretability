import json
import torch
import argparse
import os
from PIL import Image, UnidentifiedImageError
from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

def initialize_model(model_name, device, quantize_type, huggingface_token=None):
    """Initialize the model, processor, and tokenizer with optional quantization."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=huggingface_token,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": huggingface_token,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
    
    if quantize_type == '4bit':
        model_kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        })
    elif quantize_type == '8bit':
        model_kwargs.update({
            "load_in_8bit": True,
            "use_cache": True
        })

    model = MllamaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, token=huggingface_token, trust_remote_code=True)

    return model, processor, tokenizer

def load_prompts(prompt_file):
    """Load the prompt configurations from a JSON file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)

def process_images(input_data, image_dir, output_file, model, processor, prompt):
    """Process each image with the model and save the outputs using chat template for input preparation."""
    output_data = []

    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])
        try:
            # 画像の読み込み
            image = Image.open(img_path).convert("RGB")

            # チャットテンプレート用のメッセージリストを作成
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            # チャットテンプレートを適用して入力テキストを生成
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            # 画像とテキストから入力テンソルの準備
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # モデル出力の生成
            output = model.generate(**inputs, max_new_tokens=50)
            answer = processor.decode(output[0], skip_special_tokens=True)

        except (FileNotFoundError, UnidentifiedImageError):
            answer = "Image not found"
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            answer = "Error processing image"

        # 出力結果を元データに追加
        item['model_out'] = answer
        output_data.append(item)

    # すべての出力結果をファイルに保存
    with open(output_file, 'w') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="Run Vision Language Model on images with chat template input.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path.")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--quantize_type", type=str, choices=['4bit', '8bit', 'half'], default="4bit", help="Quantization type to use.")
    parser.add_argument("--huggingface_token", type=str, default=None, help="Hugging Face token for private model access.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file with question and answer sets.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--prompt_type", type=str, required=True, choices=["protanopia", "deuteranopia", "tritanopia", "normal"], help="Type of prompt to use.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the JSON file containing prompts.")

    args = parser.parse_args()

    # モデルとプロセッサの初期化
    model, processor, tokenizer = initialize_model(args.model_name, args.device, args.quantize_type, args.huggingface_token)

    #debug
    print(model.config)

    # プロンプトの読み込みと選択
    prompts = load_prompts(args.prompt_file)
    prompt = prompts[args.prompt_type]

    # 入力データのロード
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    # 画像を処理し、出力結果を生成
    process_images(input_data, args.image_dir, args.output_file, model, processor, prompt)
