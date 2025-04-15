import torch
import json
import argparse
import os
import ast
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
from huggingface_hub import login
from utils import *
seed_everything(42)

def initialize_model(model_name, attn_impl, torch_dtype, device, hf_token):
    print("DEBUG: Logging in with HF token")
    login(hf_token)

    print(f"DEBUG: Loading config for model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        token=hf_token
    ).to(device).eval()
    print("DEBUG: Model loaded and moved to device:", device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    print("DEBUG: Tokenizer loaded")
    
    # モデル独自の processor を初期化
    processor = model.init_processor(tokenizer)
    print("DEBUG: Processor initialized")
    
    return model, processor, tokenizer

def load_prompts(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

def process_images(input_data, image_dir, output_file, model, processor, tokenizer, prompt, device, prompt_type):
    # 各入力アイテムに生成結果（model_out）を追加して更新する
    output_data = []
    for item in input_data:
        img_path = os.path.join(image_dir, item['image_path'])
        # try:
        image = Image.open(img_path).convert("RGB")

        # ユーザーメッセージに prompt を組み込む
        messages = [
            {"role": "user", "content": f"<|image|>\n{prompt}"},
            {"role": "assistant", "content": ""}
        ]
        inputs = processor(messages, images=[image], videos=None).to(device)

        output_norm = model.language_model.model.norm
        unembedding_matrix = model.language_model.lm_head.weight.T

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
            print(f"Layer {layer+1} / {len(output_saving)}")
            n_answer_idx = inputs["input_ids"].shape[1] - 1
            n_answer_hidden = output[0, n_answer_idx]
            n_answer_hidden = output_norm(n_answer_hidden)
            n_answer_logits = n_answer_hidden @ unembedding_matrix
            n_answer_logprob = torch.softmax(n_answer_logits, dim=-1)
            n_answer_prob = torch.max(n_answer_logprob).item()
            n_answer_token = processor.tokenizer.decode(n_answer_logits.argmax().item())
            print(f"Answer token: {n_answer_token}, prob: {n_answer_prob:.4f}")

            n_pls_1_txt = [
                {"role": "user", "content": f"<|image|>\n{prompt}{n_answer_token}"},
                {"role": "assistant", "content": ""}
            ]

            n_pls_1_inputs = processor(
                messages=n_pls_1_txt,
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

        item["model_out"] = answer
        item["layer_outputs"] = layer_outputs
        logitlens_dir = f'/cl/home2/shintaro/color-palette-interpretability/logitlens.number.mplug'
        os.makedirs(logitlens_dir, exist_ok=True)
        logitlens_save_file = os.path.join(logitlens_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_logitlens.{prompt_type}.png")
        visualize_logitlens_from_item(item, logitlens_save_file)
        item['logitlens'] = logitlens_save_file
        output_data.append(item)

    print("DEBUG: Writing output to file:", output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
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
    parser.add_argument("--prompt_type", type=str, required=True, choices=["protanopia", "deuteranopia", "tritanopia", "normal"], help="使用するプロンプトの種類")
    parser.add_argument("--prompt_file", type=str, required=True, help="プロンプト設定が記載されたJSONファイル")
    parser.add_argument("--hf_token", type=str, required=True, help="Huggingface Access Token")

    args = parser.parse_args()

    dtype = torch.half if args.torch_dtype == "half" else torch.bfloat16
    device = args.device

    model, processor, tokenizer = initialize_model(args.model_name, args.attn_impl, dtype, device, args.hf_token)

    prompts = load_prompts(args.prompt_file)
    prompt = prompts[args.prompt_type]

    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    process_images(input_data, args.image_dir, args.output_file, model, processor, tokenizer, prompt, device, args.prompt_type)
