import os
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../Utils'))
from llm_utils import response_to_str

set_debug(True)
load_dotenv()

gemini_model = os.getenv("GEMINI_API_MODEL")
llm = ChatGoogleGenerativeAI(model=gemini_model)

# ⚠️ 强制定义好允许 Agent 操作的安全目录 (比如只允许在当前 myagent 文件夹下活动)
ALLOWED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 核心安全逻辑：防止目录穿越攻击 ("../")
def is_safe_path(file_path: str) -> bool:
    """检查文件路径是否在允许的目录下"""
    full_path = os.path.abspath(file_path)
    return full_path.startswith(ALLOWED_DIR)

@tool
def read_local_file(file_path: str) -> str:
    """这是一个读取本地源代码文件的工具。传入相对于项目根目录的文件路径。"""
    if not is_safe_path(file_path):
        return "Error: Path is not allowed."
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"
        
@tool
def write_local_file(file_path: str, content: str) -> str:
    """这是一个覆写本地文件的工具。如果你修改了代码，调用此工具将其保存到物理硬盘上。"""
    if not is_safe_path(file_path):
        return "Error: Path is not allowed."
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: Wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error: {str(e)}"

# 绑定工具
tools = [read_local_file, write_local_file]
llm_with_tools = llm.bind_tools(tools)

# 1. 定义状态 (MessagesState 等价物)
# 我们只需要维护一个核心属性：消息流 (利用 add_messages 进行状态叠加而不是覆盖)
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义大模型思考节点
def llm_node(state: GraphState):
    # 将目前为止所有的履历/黑板记录传递给 LLM
    response = llm_with_tools.invoke(state["messages"])
    # 把它自己新说的话（或者要调用工具的请求）追加进状态流
    return {"messages": [response]}

# 3. 定义路由裁判：检查 LLM 是要收工，还是要用工具
def route_after_llm(state: GraphState) -> str:
    last_message = state["messages"][-1]
    # 如果它的回答里带有 tool_calls意图，证明它要去执行动作
    if last_message.tool_calls:
        return "goto_tools"
    # 如果没带，说明它已经得出最终结论，流程可以结束
    return "end"

# 4. 手工构建我们的图
builder = StateGraph(GraphState)

# 录入 LLM 大脑节点
builder.add_node("llm", llm_node)
# 录入 工具执行 节点！(这里借用了 LangGraph 官方写好的底层执行器 ToolNode 代劳，否则你要手写复杂的 JSON 解析)
builder.add_node("tools", ToolNode(tools))
# 开始建图
builder.add_edge(START, "llm")

# 注意：当 LLM 思考完，由路由节点决定去哪里
builder.add_conditional_edges(
    "llm", 
    route_after_llm, 
    {
        "goto_tools": "tools",  # 如果要查资料/写代码，就去 tools 节点
        "end": END              # 没事了，直接终结任务
    }
)

# 当工具真的执行完了呢？必须强行回到 llm 重新思考
builder.add_edge("tools", "llm")

# 编译成引擎
agent_executor = builder.compile()

if __name__ == "__main__":
    task_prompt = "请读取当前目录下的 test_target.py，理解它的函数逻辑，然后在所有函数的注释里加上作者信息 'Author: AI Agent'，最后不要做口头解释，直接覆盖修改保存这个文件。"
    
    print(f"🚀 [任务开始]: {task_prompt}\n")
    
    # 执行图流转
    # ⚠️【高级安全防御】添加 recursion_limit 能够在一行代码中强行斩断死循环！
    # 即使大模型智商掉线、执着地不断重试报错的工具，一旦循环超过这个步数，LangGraph 就会强制抛出报错中断引擎，为你保下昂贵的 API Token！
    final_state = agent_executor.invoke({
        "messages": [HumanMessage(content=task_prompt)]
    }, config={"recursion_limit": 5})

    print("\n🏁 [任务结束] 最终答复：")
    print(final_state["messages"][-1].content)