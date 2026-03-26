import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.globals import set_debug

set_debug(False)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')

# ==========================================
# 步骤 1: 定义状态 (State) 
# 这是 LangGraph 的核心：所有的节点之间都是通过传递这个状态字典来通信的
# `add_messages` 修饰符意味着新消息会被追加到列表中，而不是覆盖
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 工具定义 (依然使用 DuckDuckGo)
search_tool = DuckDuckGoSearchRun(name="web_search", description="搜索互联网")
tools = [search_tool]

# ==========================================
# 步骤 2: 定义节点 (Nodes)
# 节点是图上的执行单元，可以是大模型推理，也可以是执行工具
# ==========================================
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
# 给大模型绑定工具说明书，赋予它输出 Tool Call JSON 的能力
llm_with_tools = llm.bind_tools(tools)

# 节点 A: 思考节点 (大模型)
def call_agent(state: AgentState):
    print("🧠 [节点执行] 大模型正在思考...")
    # 把当前所有的对话历史喂给大模型
    response = llm_with_tools.invoke(state["messages"])
    # 返回新的消息更新 State
    return {"messages": [response]}

# 节点 B: 工具执行节点
# LangGraph 提供了一个现成的 ToolNode 来帮我们批量执行大模型请求的工具
tool_node = ToolNode(tools=tools)

# ==========================================
# 步骤 3: 连线画图 (Edges)
# 定义状态图的流转规则
# ==========================================
print("🏗️ 正在编排 LangGraph 状态图...")
workflow = StateGraph(AgentState)

# 把节点挂载到图板上
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)

# 设置入口点：一开始就进入大模型思考
workflow.add_edge(START, "agent")

# 设置条件边：大模型思考完之后，下一步去哪里？
# tools_condition 内部逻辑很简单：
# 如果大模型输出了 tool_calls -> 返回 "tools" 节点
# 如果大模型正常说话 -> 返回 END (结束)
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# 当工具执行完毕后，必须强制把结果传回给大模型再次思考
workflow.add_edge("tools", "agent")

# 将图编译成可执行的程序
graph = workflow.compile()

# ==========================================
# 步骤 4: 执行图 (Graph Execution)
# ==========================================
print("\\n============ 开始 LangGraph 循环 ============")
question = "用鸭子搜索引擎查一下，目前世界上最高的建筑叫什么，位于哪个国家？"
print(f"👤 Boss: {question}\\n")

# 调用 invoke 执行图
final_state = graph.invoke(
    {"messages": [("user", question)]}
)

print("\\n" + "="*50)
print("🎉 最终任务完成报告：")
print(final_state["messages"][-1].content)
print("="*50)
