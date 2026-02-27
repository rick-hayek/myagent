import os
from dotenv import load_dotenv
from google import genai

# 1. 自动寻找并加载环境中的 .env 文件
load_dotenv()

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# 2. 实例化官方无头 SDK 客户端
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print(f"User input: Explain how AI works in a few words")

# 3. 调用底层 API 生成内容
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

# 4. 打印纯文本结果
print(f"AI response: {response.text}")