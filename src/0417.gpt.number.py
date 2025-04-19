import os
import json
import base64
import time
import argparse
from pathlib import Path
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

VISION_MODEL = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT_NAME", "gpt-4o")

def run_vision_chat(system_prompt: str, text_prompt: str, image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }
    text_block = {"type": "text", "text": text_prompt}
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": [text_block, image_block]}
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            time.sleep(10)
            return run_vision_chat(system_prompt, text_prompt, image_bytes)
        else:
            print("Error:", e)
            return None

def process_images(input_data, image_dir: str, output_file: str, prompt_config: dict):
    results = []
    for item in tqdm(input_data, total=len(input_data), desc="Processing images"):
        img_path = Path(image_dir) / item["image_path"]
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        out = run_vision_chat(
            prompt_config["system_prompt"],
            prompt_config["user_prompt"],
            img_bytes
        )
        item["model_out"] = out
        item["system_prompt"] = prompt_config["system_prompt"]
        item["user_prompt"] = prompt_config["user_prompt"]
        results.append(item)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GPT-4 with Vision on images + Q&A pairs"
    )
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file",required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=["protanopia", "deuteranopia", "tritanopia", "normal"]
    )
    args = parser.parse_args()
    with open(args.prompt_file, encoding="utf-8") as f:
        prompts = json.load(f)
    prompt = prompts[args.prompt_type]

    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    process_images(data, args.image_dir, args.output_file, prompt)
    print("Done writing to", args.output_file)
