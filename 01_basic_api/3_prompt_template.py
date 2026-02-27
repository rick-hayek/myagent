import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 创建一个 LLM 实例
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. 创建一个聊天提示词模板 (ChatPromptTemplate)
# 这是一个非常经典的双重结构：
# - system: 给 AI 设定人设、背景规则 (它不会轻易改变)
# - user: 用户的实际提问 (这是一个占位符变量 {user_input})
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的英语翻译官，并且深谙中国互联网黑话。你的任务是把用户输入的中文翻译成地道、前卫的英文。只输出翻译结果，不要带任何多余解释。"),
    ("user", "给我翻译这句话：{user_input}"),
])

# 3. 把 Prompt 模板和大模型“组合”在一起
# (注意这个 | 符号，它是 LangChain 里经典的 LCEL 管道符语法，代表把左边处理的结果传给右边)
chain = prompt | llm

print("请输入你想翻译的中文互联网黑话 (如: 我们需要一个抓手来打通底层逻辑):")
user_text = input("> ")

print("\n正在生成高大上的英文翻译...")
# 4. 执行这套组合逻辑 (在 invoke 中传入字典来替换刚才的 {user_input} 变量)
response = chain.invoke({"user_input": user_text})

print("\n[最终翻译结果]:")
print(response.content)