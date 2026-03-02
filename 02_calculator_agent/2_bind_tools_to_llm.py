import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

print(os.getenv("GEMINI_API_KEY"))

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

# 测试工具之间有依赖关系时，"并行工具调用 (Parallel Tool Calling)" 的能力
query3 = "请立刻计算出这个算式的结果：123*456+25"
print(f"\n[用户问题]: {query3}\n")

response3 = llm_with_tools.invoke(query3)

print("\n[大模型返回的文本内容 response3.content]:", response3.content)
print("\n[大模型返回的工具调用 response3.tool_calls]:\n", response3.tool_calls)
print("\n[大模型要求调用的工具列表 (tool_calls)]:")
# 只有乘法返回了，加法没有
for tool_call in response3.tool_calls:
    print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")

# 测试没有定义减法工具的调用
# 观察结果是哪种：
# 1. 自己算出结果直接回答，不调用工具
# 2. 尝试用 add(25, -13) 变通解决
# 3. 告诉你它没有减法工具
query4 = "帮我同时算两件事：第一，苹果单价3元买456个要多少钱；第二，我有25个橙子送给朋友13个还剩多少个"
print(f"\n[用户问题]: {query4}\n")

response4 = llm_with_tools.invoke(query4)

print("\n[大模型返回的完整对象 response4]:\n", response4)
print("\n[大模型返回的文本内容 response4.content]:", response4.content)
print("\n[大模型返回的工具调用 response4.tool_calls]:\n", response4.tool_calls)
print("\n[大模型要求调用的工具列表 (tool_calls)]:")
for tool_call in response4.tool_calls:
    print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")