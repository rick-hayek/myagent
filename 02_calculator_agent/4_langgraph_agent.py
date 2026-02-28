import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
# 引入 LangChain 最先进的预置 Agent 循环引擎
from langchain.agents import create_agent

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。"""
    print(f"\n[🚀 框架后台自动运行工具 multiply: a={a}, b={b}]")
    return a * b
@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    print(f"\n[🚀 框架后台自动运行工具 add: a={a}, b={b}]")
    return a + b

tools = [multiply, add]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1. 见证奇迹的时刻：把大模型和工具列表打包交给主管引擎
# create_agent 在后台帮我们写好了无比完备的 while 循环和 tool_message 拼接！
agent_executor = create_agent(llm, tools)

# 2. 我们只需要像面对一个普通模型那样，给它扔一句话就行了
query = "先计算123乘以456等于多少？算出结果后，把它加上25。"
print(f"[用户原始问题]: {query}\n")

# 3. 触发主管引擎运作
print("正在引擎内全自动多轮激战中，请观察打印日志...")
response_state = agent_executor.invoke({
    "messages": [("user", query)]
})

# 4. 直接获取它循环到底、直到完全算完后得出的最终结论！
# 框架返回的 state 中包含了所有的流转信息，最后一条 message 就是彻底完工的自然语言回复
final_response = response_state["messages"][-1]
print("\n🎉【大模型的最终人类自然语言回复】:\n", final_response.content)