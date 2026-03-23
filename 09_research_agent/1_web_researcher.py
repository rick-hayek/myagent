import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic import hub
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')
print(f"GEMINI_API_MODEL: {gemini_model}")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# ==========================================
# 步骤 1: 组装你的网络研究员工具箱
# ==========================================
print("🔧 正在初始化 DuckDuckGo 搜索引擎工具...")
# DuckDuckGo 免费且不需要申请 API Key，最适合拿来练手！
search_tool = DuckDuckGoSearchRun()

# 把搜索工具装进数组
tools = [search_tool]

# ==========================================
# 步骤 2: 构建超现代的 Tool Calling Agent
# ==========================================
print("🤖 正在唤醒具备原生 Tool Calling 能力的 Gemini 大模型...")
# 注意：并不是所有模型都原生支持 Tool Calling。Gemini 是深度原生支持这套 JSON 格式交互的。
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)

# 拉取最现代的 Tool Calling 专属 Prompt (替代了老掉牙的 hwchase17/react)
# 它不再强迫模型输出 Thought/Action 这种文本，而是让底层直接走 API
prompt = hub.pull("hwchase17/openai-functions-agent")
# 我们使用了 create_tool_calling_agent 而不是 create_react_agent！
agent = create_tool_calling_agent(llm, tools, prompt)

# 依然装进包装器里，由包装器负责捕获 JSON 并实际调用 Python 函数
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# ==========================================
# 步骤 3: 释放你的猛兽！
# ==========================================
print("\n============ 开始全网搜索与研究 ============")
# 下达一个必须分治、联网、并且综合分析的顶级复杂任务！
complex_task = "谁是2024年诺贝尔文学奖的获得者？请用50字概括一下他/她的主要写作风格或代表作特点。"

print(f"👤 Boss 提问: {complex_task}")
response = agent_executor.invoke({"input": complex_task})

print("\n" + "="*50)
print("🎉 最终任务完成报告：")
print(response['output'])
print("="*50)