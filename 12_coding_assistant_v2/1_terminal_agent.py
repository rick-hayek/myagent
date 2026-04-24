import os
from dotenv import load_dotenv
from asyncio import subprocess
import subprocess
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.globals import set_debug
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

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
def execute_terminal_command(command: str) -> str:
    """终极终端工具。用于执行 bash 命令，比如编译、测试代码等。"""
    try:
        # 注意：在生产环境不可将 shell=True 直接暴露在外，这里仅作本地沙箱教学
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        output = result.stdout
        if result.stderr:
            output += f"\n[Error]: \n{result.stderr}"
        
        return output if output else "Command executed successfully with no output."
    
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """读取本地文件"""
    try:
        if not is_safe_path(file_path):
            return "Error: Access denied. You are not allowed to read files outside the project directory."
        if not os.path.exists(file_path):
            return f"read_local_file: Error: File not found: {file_path}"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"read_local_file: Error: {str(e)}"

@tool
def write_local_file(file_path: str, content: str) -> str:
    """覆写本地文件"""
    try:
        if not is_safe_path(file_path):
            return "Error: Access denied. You are not allowed to write files outside the project directory."
        if not os.path.exists(file_path):
            return f"write_local_file: Error: File not found: {file_path}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"write_local_file: Error: {str(e)}"


tools = [execute_terminal_command, read_local_file, write_local_file]
llm_with_tools = llm.bind_tools(tools)

class StateMessage(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # 我们删除了手工的 error_count，因为在用 Messages 流的图里，一切都在聊天记录里！

def llm_node(state: StateMessage):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def route_after_llm(state: StateMessage):
    # 🌟【优雅解法】直接从图的对话记忆中动态计算！
    error_msg_count = sum(
        1 for msg in state["messages"] 
        if isinstance(msg, ToolMessage) and "Error:" in msg.content
    )
    
    if error_msg_count >= 3:
        # 当超过阈值，返回触发人工节点的路由指令
        return "human_in_loop"
        
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "goto_tools"
    return "end"

# 1. 新增一个空壳节点，唯一的作用是为了让图能流转到这里停下（挂起等待人类）
def ask_human_node(state: StateMessage):
    """人工介入节点。图执行到这里前会被强行挂起。"""
    pass

builder = StateGraph(StateMessage)
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("ask_human", ask_human_node) # 注册人工节点

builder.add_edge(START, "llm")
builder.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "goto_tools": "tools",
        "human_in_loop": "ask_human", # 正确映射到刚注册的 ask_human 节点
        "end": END
    })

# 2. 补上后半段的边
builder.add_edge("tools", "llm")
builder.add_edge("ask_human", "llm") # 当人类介入并修改完信息后，必须引导流程回到大模型

memory = MemorySaver()
app = builder.compile(
    checkpointer=memory,
    # 3. 核心断点配置：不仅在做危险执行(tools)前要停下，遇到过量报错求助人类(ask_human)时更要停下！
    interrupt_before=["tools", "ask_human"]
)

if __name__ == "__main__":
    # 配置必须传入固定的线程 ID 才能激活记忆
    config = {"configurable": {"thread_id": "hack_session_1"}}
    
    # 向系统下达可能需要终端执行的指令
    user_input = "请帮我在当前目录跑一下 python -c 'print(\"Hello Terminal\")'"

    for event in app.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        last_message.pretty_print()
    
    # == 第一阶段图执行完毕，拉起了手刹 ==
    # 大模型已经决议好要调用工具了，现在图卡在了 interrupt_before=["tools"]
    snapshot = app.get_state(config)
    next_node = snapshot.next 
    
    print(f"\n[系统检查]: 下一个节点是: {next_node}")
    if next_node and "tools" in next_node:
        print("\n🚨 [安全警告] AI 正试图执行物理工具（读写或终端命令）！")
        # 你甚至可以从 snapshot.values 里剥析出它到底要执行什么 shell 语句
        print(f"\n AI 想要执行的命令是: \n {snapshot.values}")
        
        user_approval = input("是否准许执行此危险动作？(y/n): ")
        if user_approval.lower() == 'y':
           print("\n⚡️ 授权通过，让工具继续执行...")
           # 关键步骤：调用 .continue() 唤醒被挂起的图
           for event in app.stream(None, config=config, stream_mode="values"):
           #for event in app.continue(config, stream_mode="values"):
               last_message = event["messages"][-1]
               last_message.pretty_print()
        else:
            print("\n❌ 拒绝执行。图已挂起，等待人工指令。")
    elif next_node and "ask_human" in next_node:
        print("\n🚨 [安全警告] AI 连续报错 3 次以上，拉起人工介入节点...")
        help_info = input("连续出错，需给新的提示: ")
        app.update_state(config, {"messages": [HumanMessage(content=help_info)]})
        for event in app.stream(None, config=config, stream_mode="values"):
            last_message = event["messages"][-1]
            last_message.pretty_print()

        
            