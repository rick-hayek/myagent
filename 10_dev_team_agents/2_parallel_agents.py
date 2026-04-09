import os
import operator
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langgraph.types import Send
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')

# 核心状态板设计：用 Annotated 与 operator.add 进行数组追加
class ParallelTeamState(TypedDict):
    task: str
    prd: str
    code_snippets: Annotated[list[str], operator.add]
    
llm = ChatGoogleGenerativeAI(model=gemini_model)

def response_to_str(response: AIMessage) -> str:
    content = response.content
    if isinstance(content, list):
        text = content[0]["text"]
    else:
        text = content
    return text

def product_manager(state: ParallelTeamState):
    print("👨💼 [PM] 正在分解微服务架构...")
    sys_msg = SystemMessage(content="你是顶级的架构师，请将需求拆解为微服务架构")
    usr_msg = HumanMessage(content=state['task'])
    response = llm.invoke([sys_msg, usr_msg])
    return {"prd": response_to_str(response)}

def frontend_engineer(prd_chunk: str):
    # 【注意！】这里通过 Send API 传过来的不是一整个大 State，而是你分配给它的一份特定的切片数据
    print(f"🎨 [Frontend] 收到专属需求，正在并发画界面...")
    sys_msg = SystemMessage(content="你是顶级的UI工程师，请根据需求输出HTML代码")
    usr_msg = HumanMessage(content=prd_chunk)
    response = llm.invoke([sys_msg, usr_msg])
    return {"code_snippets": ["<h1>这是前端代码</h1>" + response_to_str(response)]}

def backend_engineer(prd_chunk: str):
    print(f"⚙️ [Backend] 收到专属需求，正在并发写后端...")
    sys_msg = SystemMessage(content="你是顶级的后端工程师，请根据需求输出Python代码")
    usr_msg = HumanMessage(content=prd_chunk)
    response = llm.invoke([sys_msg, usr_msg])
    return {"code_snippets": ["<h1>这是后端代码</h1>" + response_to_str(response)]}

# 这是一个路由决策函数
def dispatch_engineer(state: ParallelTeamState):
    print("🚦 [Router] 需求已出！正在将任务【同时 Send】给前端和后端！")
    prd = state['prd']
    
    return [
        Send("frontend_engineer", prd),
        Send("backend_engineer", prd)
    ]
    
# 状态图构建
builder = StateGraph(ParallelTeamState)

# 添加节点
builder.add_node("pm", product_manager)
builder.add_node("frontend_engineer", frontend_engineer)
builder.add_node("backend_engineer", backend_engineer)

# 启动流程：从 PM 开始
builder.add_edge(START, "pm")

# 重点！用条件边把 PM 和 路由器 连起来。
# 这里不用 add_edge，而是把 dispatch_engineers 返回的 Send 对象，交给 LangGraph 的底层引擎去扇出 (Fan-out) 激活对应的节点
builder.add_conditional_edges(
    "pm",
    dispatch_engineer,
    ["frontend_engineer", "backend_engineer"]
)

# 工程师干完活，归拢到终点 (Fan-in)
builder.add_edge("frontend_engineer", END)
builder.add_edge("backend_engineer", END)

app = builder.compile()

if __name__ == "__main__":
    final_state = app.invoke({
        "task": "写一个登录注册的微服务功能", 
        "code_snippets": []
    })
    
    # 看看最终聚合的数组里，是不是同时包含了两份代码？
    print("\n📦 全部代码交付件：")
    for idx,code in enumerate(final_state['code_snippets']):
        print(f"\n--- 代码块 {idx+1} ---")
        print(code)
        
