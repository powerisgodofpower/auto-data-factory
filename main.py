import os
import google.generativeai as genai
from huggingface_hub import HfApi, login

# 1. 鍵を取り出す
gemini_key = os.environ.get("GEMINI_API_KEY")
hf_token = os.environ.get("HF_TOKEN")

# 2. Geminiの準備（AIの脳みそ）
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# 3. データ生成（Geminiに考えてもらう：今回は豆知識）
response = model.generate_content("AIやテクノロジーに関する面白い豆知識を1つ教えてください。")
data_text = response.text
print("生成されたデータ:", data_text)

# 4. データをファイルに追記して保存
with open("ai_knowledge.txt", "a", encoding="utf-8") as f:
    f.write(data_text + "\n---\n")

# 5. Hugging Faceへアップロード（出品）
login(token=hf_token)
api = HfApi()

# ★注意：ここの 'YOUR_HF_USERNAME' を自分のHugging Faceのユーザー名に変える！★
HF_REPO = "powerisgodofpower/auto-generated-data" 

# データセットの箱がなければ作る
try:
    api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)
except Exception as e:
    pass

# データをアップロード
api.upload_file(
    path_or_fileobj="ai_knowledge.txt",
    path_in_repo="ai_knowledge.txt",
    repo_id=HF_REPO,
    repo_type="dataset"
)
print("Hugging Faceへのアップロード完了！")
