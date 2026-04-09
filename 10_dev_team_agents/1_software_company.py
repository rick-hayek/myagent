import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.globals import set_debug
from langchain_core.tools import tool

set_debug(True)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')
# 整个公司共用一个大脑，但待会给他们套上不同的外衣（System Prompt）
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.2)

# ==========================================
# 步骤 1: 创办公司，定义公司的“黑板” State
# 三个员工都在这块黑板上读写数据
# ==========================================
class TeamState(TypedDict):
    task: str           # 老板的任务
    prd: str            # 产品经理写的需求文档PRD
    code: str           # 程序员写的代码
    feedback: str       # 测试员给的反馈
    committer: str      # 提交代码
    iterations: int     # 迭代次数: 为了防止测试员和程序员无限吵架，加个计数器

# ==========================================
# 步骤 2: 招募员工 (定义 Nodes)
# ==========================================

# PM: 输入老板需求，输出极简的、条理清晰、需求明确的PRD
def product_manager(state: TeamState):
    print("👨💼 [PM] 收到老板需求，正在拆解 PRD...")
    sys_msg = SystemMessage(content="你是顶级的互联网产品经理。你的任务是将传入的抽象『老板需求』，拆解为一份极简的、条理清晰的『产品体验文档 (PRD)』。不用太长，列出核心功能点即可。")
    user_msg = HumanMessage(content=state['task'])
    response = llm.invoke([sys_msg, user_msg])
    return {"prd": response.content}

# 牛马程序员：输入PRD，输出代码
def coder(state: TeamState):
    print("🧑‍💻 [Coder] 收到PRD，正在疯狂敲代码...")
    prompt = f"这是产品经理的 PRD:\n{state['prd']}\n\n"  # 始终把PRD放在最前面，让AI知道核心需求
    if state.get('feedback'):
        # 如果有QA的反馈，把反馈加到prompt里
        print("🧑‍💻 [Coder] （看到测试反馈了，正在修复 Bug）...")
        prompt += f"这是测试员打回来的反馈意见：\n{state['feedback']}\n请根据反馈修改代码！\n"
    
    prompt += "仅仅输出Python代码即可，不要废话。"

    sys_msg = SystemMessage(content="你是顶配全栈工程师。仔细看需求，写出健壮漂亮的Python代码。")
    user_msg = HumanMessage(content=prompt)
    response = llm.invoke([sys_msg, user_msg])
    return {"code": response.content}

# 牛马测试员：输入PRD和代码，输出反馈
def reviewer(state: TeamState):
    print("🕵️‍♂️ [Reviewer] 拿到代码，正在做严格的代码审查...")
    prompt = f"PRD 需求是:\n{state['prd']}\n\n程序员写的代码是:\n{state['code']}\n"
    prompt += "如果代码完美实现了所有需求且没有明显 Bug，请严格原样输出字符串 'PASS'。\n如果存在漏掉的功能或致命逻辑漏洞，请指出具体问题作为 Feedback 返回。"
    
    sys_msg = SystemMessage(content="你是最冷酷无情的资深架构师测试员。你的眼里容不得半粒沙子。对代码进行批判性评审。")
    user_msg = HumanMessage(content=prompt)
    response = llm.invoke([sys_msg, user_msg])
    
    iters = state.get("iterations", 0) + 1
    return {"feedback": response.content, "iterations": iters}

def write_final_code_to_file(filename: str, code: str) -> bool:
    # 验证filename合法性与安全性，限制只能在当前目录下生成文件
    if ".." in filename or filename.startswith("/"):
        return False
    
    # 写入最终代码文件
    save_path = os.path.join(os.path.dirname(__file__), filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(code)
    return True

import re

# 提交代码：PASS之后生成最终代码文件
def committer(state: TeamState):
    print("📦 [Committer] 代码已通过审查，正在打包成最终交付件...")
    code = state['code']
    if(isinstance(code, type([]))):
        final_code = code[0]["text"]
    else:
        final_code = code

    # 使用正则剥离出 ```python 里的纯代码
    match = re.search(r"```python\n(.*?)\n```", final_code, re.DOTALL)
    if match:
        final_code = match.group(1)
    else:
        # 万一大模型直接给了纯代码，也做一下简单的去首尾空白处理
        final_code = final_code.strip()

    # 提取文件名
    filename = "1_software_company_generated_snake_game.py"
    if write_final_code_to_file(filename, final_code):
        #return {"committer": "代码已保存到: " + filename}
        return {"code": final_code} # 顺便覆盖重写原来带 ``` 的代码

# ==========================================
# 步骤 3: 制定公司流程审批制度 (Edges)
# ==========================================
print("🏗️ 正在注册公司 (构建 LangGraph)...")
builder = StateGraph(TeamState)

# 录入三个岗位
builder.add_node("product_manager", product_manager)
builder.add_node("coder", coder)
builder.add_node("reviewer", reviewer)
builder.add_node("committer", committer)

builder.add_edge(START, "product_manager")
builder.add_edge("product_manager", "coder")
builder.add_edge("coder", "reviewer")
# builder.add_edge("reviewer", "committer") # 🚨 移除该无条件边，因为下方已经定义了条件边
builder.add_edge("committer", END)

# 核心条件流转：测试员看完之后怎么办？
def reviewer_decision(state: TeamState) -> str:
    # 强制跳出：避免两牛马吵架超过 3 次导致公司破产
    if state.get("iterations", 0) >= 3:
        print("🚦 [系统] 团队争论达到上限 3 次，强制交付当前版本。")
        return "end"
    
    if "PASS" in state.get("feedback", ""):
        print("✅ [系统] 测试员打出 PASS！代码完美通过审查，交付客户！")
        return "end"
    else:
        print("❌ [系统] 测试员要求返工！工单退回给程序员...")
        return "continue"

# 根据测试决定流向
builder.add_conditional_edges(
    "reviewer",
    reviewer_decision,
    {
        "end": "committer",
        "continue": "coder"
    }
)

app = builder.compile()

# ==========================================
# 步骤 4: 公司开张运营拉客！
# ==========================================
if __name__ == "__main__":
    task = "写一个简易命令行贪吃蛇游戏。必须有计分功能，撞墙或者撞自己就会死。"
    print(f"\\n💼 [Boss 任务下达]: {task}\\n")
    
    # 注入公司第一笔启动资金和活儿
    initial_state = {"task": task, "iterations": 0}
    
    # 执行图引擎
    # 我们不直接取最后一个 state，直接使用打印日志即可看出流程
    final_output = app.invoke(initial_state)
    
    print("\\n" + "="*50)
    print("🏢 软件产品最终交付:")
    print(final_output["code"])
    print("="*50)
