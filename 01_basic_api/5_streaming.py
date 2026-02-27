import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. 创建 LLM 实例
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. 创建 Prompt 模板
prompt = ChatPromptTemplate.from_template("请帮我写一首关于{topic}的长诗，至少包含四段。")

# 3. 组装我们熟悉的核心三剑客链路：Prompt -> LLM -> 提取纯文本的 Parser
chain = prompt | llm | StrOutputParser()

# 输入主题
topic_input = input("请输入你想写诗的主题: ")
print("\n[AI 正在努力作诗中，请欣赏终端里的打字机效果]...\n")

# 4. 【核心差异】不要用 invoke()，而是使用 stream() 方法！
# stream() 方法会返回一个可以被迭代的流式生成器 (Generator)
# 只要大模型有哪怕一个新的字符吐出来，这个 for 循环内部的代码就会立刻被触发一次
for chunk in chain.stream({"topic": topic_input}):
    # print 默认会在每个字符后加换行，我们要改成加空字符串，并强制立刻刷新输出流展示到屏幕(flush=True)
    print(chunk, end="", flush=True)

print("\n\n[诗歌创作完毕！]")