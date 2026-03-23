import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_classic import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.globals import set_debug

set_debug(False)
load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')
print(f"GEMINI_API_MODEL: {gemini_model}")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# ==========================================
# 工具 1: 眼睛 (搜索引擎)
# ==========================================
search_tool = DuckDuckGoSearchRun(name="web_search", description="用于搜索最新的互联网实时信息。")

# ==========================================
# 工具 2: 手 (本地文件写入器)
# ==========================================
# 注意：赋予 AI 操作本地文件的权限，这就是数字员工干活的开端！
@tool
def write_report_to_file(filename: str, content: str) -> str:
    """当需要保存工作成果或输出报告时，必须调用此工具将内容写入Markdown文件中。
    filename: 文件名（必须以.md结尾，直接写文件名即可）
    content: 你要写入的具体报告内容长文
    """
    save_path = os.path.join(os.path.dirname(__file__), filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"【系统回执】太棒了！你的长篇报告已成功写入本地硬盘：{save_path}"

# 组装混合工具箱囊：能看，能写
tools = [search_tool, write_report_to_file]

llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.3)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)

# ==========================================
# 终极任务下达
# ==========================================
print("\n🔥 [Boss 下达指令] Agent 已启动，这需要它跑动多个流程：思考 -> 搜索 -> 总结 -> 写入文件")
hard_task = """
请帮我研究一下最近很火的 "DeepSeek R1" 大模型。
1. 去网上搜索它的核心亮点或独特架构。
2. 将研究结果整理成一篇层次分明的 Markdown 迷你报告。
3. 调用写文件工具，将报告保存到文件名为 `deepseek_r1_report.md` 的文件中。
4. 确保文件写入成功后，告诉我一句 "报告长官，任务完成！" 即可。
"""

agent_executor.invoke({"input": hard_task})
print("\n任务流程结束。快看看项目左侧 09_research_agent 目录下有没有多出一个文件吧！")
    