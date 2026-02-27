import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 加载环境变量中的 API Key
load_dotenv()

# 2. 实例化 LangChain 包装的大模型对象
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
print("通过LangChain调用Google Gemini API")
print(f"input: Explain how AI works in a few words")

# 3. 调用 invoke 方法向模型提出问题
response = llm.invoke("Explain how AI works in a few words")

# 4. 解析并打印回答内容。注意 response 是一个 AIMessage 对象。
print("\n [LangChain 返回的完整对象类型]:", type(response))
print("\n [AI 生成的文本内容]:", response.content)

