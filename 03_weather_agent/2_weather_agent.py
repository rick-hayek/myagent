import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
# 只要这样导入一下刚才写的工具函数 (因为文件名以数字开头，必须用 importlib 动态绕过限制)
# from 1_weather_tool import get_weather
import importlib
weather_module = importlib.import_module("1_weather_tool")
get_weather = weather_module.get_weather

load_dotenv()

# 1. 注册工具箱
tools = [get_weather]

# 2. 召唤大模型
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. 组装智能体引擎
agent_executor = create_agent(llm, tools)

# 4. 来一句典型的需要调用天气工具的问题
query = "我明天打算去北京出差，请帮我查一下那里现在的天气情况，并根据情况给我一些穿衣建议。"
print(f"[用户原始问题]: {query}\n")

print("正在引擎内全自动多轮激战中，请观察打印日志...")

# 5. 启动智能体引擎
response_state = agent_executor.invoke({
    "messages": [("user", query)]
})

# 6. 提取最终结果
final_response = response_state["messages"][-1]
# print("\n🎉【大模型的最终人类自然语言回复】:\n", final_response.content) # 返回可能是json数据也可能是处理好的自然语言数据
# 优雅处理
content = final_response.content
if isinstance(content, list):
    # 如果它是列表，寻找 type 为 text 的那一块取出来
    print("content is list")
    final_text = "".join(block["text"] for block in content if block.get("type") == "text")
else:
    # 如果 LangChain 帮我们把它自动折叠成了纯字符串，那就直接用！
    print("content is not list")
    final_text = content

print("\n🎉【大模型的最终人类自然语言回复】:\n", final_text)