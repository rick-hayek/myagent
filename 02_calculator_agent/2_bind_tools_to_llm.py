import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。
    Args:
        a: 第一个被乘数
        b: 第二个乘数
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    return a + b

tools = [multiply, add]

# 1. 正常初始化大模型
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. 【核心魔法】将工具绑定给大模型！
# 这行代码会在底层把 tools 的 JSON Schema 说明书跟 LLM 进行强绑定。
llm_with_tools = llm.bind_tools(tools)

# 3. 准备一个需要多步计算的复杂问题
query = "计算123乘以456等于多少？然后再把25加上13算算等于多少？"
print(f"[用户问题]: {query}\n")

# 4. 调用绑定了工具的全新大模型
response = llm_with_tools.invoke(query)

# 5. 观察奇迹发生：拆解大模型返回的 AIMessage 对象
print("\n[大模型返回的完整对象 response]:\n", response)
print("\n[大模型返回的文本内容 response.content]:", response.content)
print("\n[大模型返回的工具调用 response.tool_calls]:\n", response.tool_calls)
print("\n[大模型要求调用的工具列表 (tool_calls)]:")
# 只有乘法返回了，加法没有
for tool_call in response.tool_calls:
    print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")

# 测试"并行工具调用 (Parallel Tool Calling)" 能力
query2 = "请同时立刻帮我做两件事：计算123*456，以及计算25+13"
print(f"\n[用户问题]: {query2}\n")

response2 = llm_with_tools.invoke(query2)

print("\n[大模型返回的文本内容 response2.content]:", response2.content)
print("\n[大模型返回的工具调用 response2.tool_calls]:\n", response2.tool_calls)
print("\n[大模型要求调用的工具列表 (tool_calls)]:")
# 两个工具乘法和加法都返回了
for tool_call in response2.tool_calls:
    print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")
