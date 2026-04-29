import os
from dotenv import load_dotenv
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

DANGEROUS_PATTERNS = ["rm -rf", "mkfs", "dd if=", ":(){ :|:& };:", "> /dev/", "curl | bash"]
def is_safe_command(command: str) -> bool:
    return not any(p in command for p in DANGEROUS_PATTERNS)

@tool
def execute_terminal_command(command: str) -> str:
    """终极终端工具。用于执行 bash 命令，比如编译、测试代码等。"""
    if not is_safe_command(command):
        return f"Error: Unsafe command detected: {command}"
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
            nearby_files = os.listdir(os.path.dirname(file_path) or '.')
            return f"read_local_file: Error: File not found: {file_path}. 当前目录下有这些文件: {nearby_files}"
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
    config = {"configurable": {"thread_id": "boss_battle_session2"}}
    
    # 最终极的闭环提示词！
    boss_prompt = """
    这是你的一项独立修复任务：
    1. 在当前目录下，直接使用终端命令运行 `pytest test_bad_math.py`。
    2. 你一定会看到测试报错（因为源文件坏了）。请不要向我解释，立刻使用读取文件工具查看 `bad_math.py` 的源码。
    3. 分析它为什么通不过那些报错的测试用例。
    4. 使用写文件工具，把修复好 Bug 的代码覆写进 `bad_math.py` 里。
    5. 再次运行 `pytest test_bad_math.py` 验证你的修复结果。
    6. 如果还是报错，继续重复步骤2-5。如果你最终看见了全绿的 PASS，就可以告诉我任务完成了！
    """
    print("\n🚀 [系统准备] 准备将任务发送给大模型，这可能需要一些时间（尤其是网络较慢时）...")
    try:
        for event in app.stream({"messages": [("user", boss_prompt)]}, config=config, stream_mode="values"):
            print("\n⚡️ [系统接收] 收到图流转的新事件！")
            last_message = event["messages"][-1]
            last_message.pretty_print()
    except Exception as e:
        print(f"\n❌ [系统崩溃] 执行时发生异常: {e}")
    
    print("\n🛑 [第一阶段结束] 准备进入循环拦截...")
    
    # == 第一阶段图执行完毕，拉起了手刹 ==
    # 大模型一旦决议要调用工具，图就会卡在 interrupt_before=["tools"]，并退出 stream
    # ⚠️ 核心修复：因为 AI 会进行【多次】反思和工具调用，我们必须用 while 循环不断拦截并授权！
    while True:
        print("STARTING WHILE LOOP")
        snapshot = app.get_state(config)
        next_node = snapshot.next 
        print(f"\n\nIN WHILE: snapshot: {snapshot}")
        print(f"\n\nIN WHILE: next_node: {next_node}")

        if not next_node:
            print("\n🎉 [图流转结束] 没有任何后续节点需要执行了。任务完成！")
            break
            
        print(f"\n[系统检查]: 下一个被挂起的节点是: {next_node}")
        
        if "tools" in next_node:
            print("\n🚨 [安全警告] AI 正试图执行物理工具（读写或终端命令）！")
            last_ai_message = snapshot.values["messages"][-1]
            if hasattr(last_ai_message, "tool_calls") and last_ai_message.tool_calls:
                for tool_call in last_ai_message.tool_calls:
                    print(f" 工具：{tool_call['name']}")
                    print(f" 参数：{tool_call['args']}")
            
            user_approval = input("是否准许执行此危险动作？(y/n) [输入 q 退出]: ")
            if user_approval.lower() == 'q':
                break
            elif user_approval.lower() == 'y':
                print("\n⚡️ 授权通过，让工具继续执行...")
                # 唤醒图，跑完工具后，它会自动流向 LLM，如果 LLM 还要跑工具，就会再次触发中断并进入下一个 while 循环
                for event in app.stream(None, config=config, stream_mode="values"):
                    last_message = event["messages"][-1]
                    last_message.pretty_print()
            else:
                print("\n❌ 拒绝执行。图已挂起，等待人工指令。")
                break
                
        elif "ask_human" in next_node:
            print("\n🚨 [安全警告] AI 连续报错 3 次以上，拉起人工介入节点...")
            help_info = input("连续出错，需给新的提示 [输入 q 退出]: ")
            if help_info.lower() == 'q':
                break
            app.update_state(config, {"messages": [HumanMessage(content=help_info)]})
            for event in app.stream(None, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                last_message.pretty_print()

        
            