import os
from datetime import datetime
import google.generativeai as genai
from huggingface_hub import HfApi

# 1. APIキーとリポジトリの設定
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = "Power2007/auto-generated-data"

# 2. GeminiのAPI設定
genai.configure(api_key=GEMINI_API_KEY)

def generate_data():
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # 3. 英語で世界中の開発者に刺さる「尖ったテーマ」を指示するプロンプト
    prompt = f"""
    Generate a high-value, structured technical dataset entry in English for AI developers and software engineers.
    Generated timestamp (UTC): {today}
    
    Topic: Advanced programming edge cases, tricky debugging solutions, and system architecture anti-patterns.
    
    Format requirements:
    - Write clearly in English.
    - Include a specific technical problem scenario.
    - Provide a concise, highly efficient code snippet or configuration fix (Python, Docker, or GitHub Actions).
    - Explain the core reason why the issue happens and how to prevent it.
    - Keep it practical, structured, and ready to be used as fine-tuning data for LLMs.
    """
    
    # 安定版のモデルを指定
    model = genai.GenerativeModel('gemini-3.8-flash')
    response = model.generate_content(prompt)
    
    return response.text

def main():
    print("Generating global-ready technical dataset via Gemini...")
    content = generate_data()
    
    filename = f"tech_insight_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # ローカルに一時保存
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Saved locally as {filename}")
    
    # 4. Hugging Faceへ自動プッシュ
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"Successfully uploaded {filename} to Hugging Face: {HF_REPO}")

if __name__ == "__main__":
    main()
