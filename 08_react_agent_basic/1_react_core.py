import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.globals import set_debug

# 开启上帝视角，这是看懂 ReAct 内心戏的关键！
set_debug(True)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')
print(f"GEMINI_API_MODEL: {gemini_model}")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# ==========================================
# 步骤 1: 打造 Agent 的武器库 (Tools)
# ==========================================
# 1. 第一个工具：一个绝对精准的乘法器
# 注意这里的 docstring("""乘法器...""")，这是告诉大模型这个工具是干嘛用的，大模型全靠这段中文注释来决定要不要抓起这个工具！
# 【重大坑点】：最古老的纯文本 ReAct 框架很难处理多个参数，所以强制要求模型只传入一个字符串，我们自己切分！
@tool
def multiply(input_str: str) -> str:
    """乘法器。当你需要计算两个数字相乘时，调用这个工具。输入必须且只能是两个用逗号隔开的数字，例如 "5, 10" """
    try:
        a, b = input_str.split(",")
        print(f"\\n🔧 [工具被拔出] 乘法器正在计算 {a.strip()} * {b.strip()} ...")
        ans = float(a) * float(b)
        return str(ans)
    except Exception as e:
        return f"输入格式错误，请务必只提供逗号隔开的两个数字，例如 '5, 10'。错误信息: {e}"

# 2. 第二个工具：本地机密信息查询系统
@tool
def get_secret_info(name: str) -> str:
    """机密库。当你需要查询某人的年纪、身份或密码等机密信息时，传入人名，调用这个工具。"""
    print(f"\\n🔧 [工具被拔出] 正在查询 {name} 的机密档案 ...")
    db = {"Rick": "30岁，有一艘银河护卫舰。", "Morty": "14岁，跟着Rick到处跑。"}
    return db.get(name, "查无此人")

# 把两个武器装进箱子
tools = [multiply, get_secret_info]

# ==========================================
# 步骤 2: 组装 ReAct 智能体 (Agent)
# ==========================================
# 选一个脑子比较好用的大模型（必须聪明到能看懂工具说明书）
llm = ChatGoogleGenerativeAI(model=gemini_model)
# 去 LangChain Hub 下载最经典的官方 ReAct 咒语模板 (hwchase17/react)
prompt = hub.pull("hwchase17/react")
# 把 大脑(llm) + 武器库(tools) + 思考说明书(prompt) 融合，锻造出 Agent 的灵魂
agent = create_react_agent(llm, tools, prompt)

# 把灵魂装进一个死循环执行引擎里，赋予它不断尝试的肉身
# max_iterations=5 是保命符！防止大模型脑抽陷入死循环无限调 API 烧钱
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# ==========================================
# 步骤 3: 下达复杂复合指令，观察它的表演！
# ==========================================
print("============ 开始执行任务 ============")
# 这是一个需要大模型自主规划两步走的复合问题：先查资料，再算数学题！
question = "Rick今年几岁了？他的年龄乘以 100 是多少？得到的结果乘以Morty的年龄是多少？"
print(f"👤 Boss 提问: {question}")
response = agent_executor.invoke({"input": question})
print("\\n" + "="*50)
print("🎉 最终任务完成报告：")
print(response['output'])
print("="*50)