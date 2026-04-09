import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

gemini_model = os.getenv('GEMINI_API_MODEL')
print(f"GEMINI_API_MODEL: {gemini_model}")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# ==========================================
# 步骤 1: 准备纯 Python 的本地基础工具
# ==========================================
def multiply(input_str: str) -> str:
    try:
        a, b = input_str.split(",")
        return str(float(a.strip()) * float(b.strip()))
    except Exception as e:
        return f"输入错误: {e}"

def get_secret_info(name: str) -> str:
    db = {"Rick": "30岁，有一艘银河护卫舰。", "Morty": "14岁，跟着Rick到处跑。"}
    return db.get(name, "查无此人")

# 手动建一个工具路由表
tool_map = {
    "multiply": multiply,
    "get_secret_info": get_secret_info
}

# ==========================================
# 步骤 2: 编写人类有史以来最伟大的 ReAct 裸 Prompt
# ==========================================
# 我们不使用任何黑盒，直接把规则写在脸上告诉大模型！
REACT_SYSTEM_PROMPT = """
尽可能按照最优逻辑回答用户的问题。
你可以使用以下工具：
- multiply: 乘法器，当你想做乘法时调用。必须输入格式如 "5, 10" 的两个数字
- get_secret_info: 机密查询，传入人名查询其机密档案，例如 "Rick"

你必须、绝对、强制要求按照以下格式返回文本给你自己做思考：

Question: 用户提出的问题
Thought: 你应该总是思考下一步要做什么
Action: 你决定使用的工具名称，必须是 [multiply, get_secret_info] 中的一个
Action Input: 传给工具的参数
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道了最终答案
Final Answer: 最终回答用户的话

开始！
"""

# ==========================================
# 步骤 3: 手写大模型外层的 While 死循环 (AgentExecutor 的底层)
# ==========================================
print("============ 开始执行纯手工 Agent 裸循环 ============")
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)

question = "Rick今年几岁了？他的年龄乘以 100 是多少？得到的结果乘以Morty的年龄是多少？"
print(f"👤 Boss 提问: {question}\n")

# 这是我们要不断往里追加记录的“大脑记事本”
chat_history = REACT_SYSTEM_PROMPT + f"\nQuestion: {question}\n"

# 设定最大循环次数为 5，防止破产
max_iterations = 10

for i in range(max_iterations):
    print(f"\n--- ⏳ 第 {i+1} 轮循环思考开始 ---")
    
    # 1. 这里是 Agent 架构最最最核心的灵魂：Stop Words (停止词)！
    # 如果不加 stop=["Observation:"], 大模型戏瘾大发，不仅会写出 Action，还会像戏精一样自己把 Observation 的结果编造出来（比如编造你70岁）。
    # 加了 stop 后，大模型一准备输出 Observation，接口就会立刻掐断它的喉咙，把控制权交还给下方的 Python 代码去真的执行本地函数！
    response = llm.bind(stop=["Observation:"]).invoke(chat_history)

    print(f"🤖 LLM 输出片段: \n{response.content}")
    content = response.content
    if(isinstance(content, list)):
        output_text = content[0]["text"]
    else:
        output_text = content
    print(f"🤖 处理后输出: \n{output_text}")
    
    # 将 LLM 刚才说的话原封不动记回到历史记事本里
    chat_history += output_text
    chat_history += "\n"
    
    # 2. 判断任务是不是完成了？
    if "Final Answer:" in output_text:
        print("\n🎉 侦测到 Final Answer！任务执行完毕跳出死循环！")
        break
        
    # 3. 如果没完成，就硬核解析 LLM 输出里的文本，提取它想干什么
    # 这里我们用最简单的 Python 字符串查找 (find) 来剥离文本
    try:
        action_idx = output_text.rindex("Action:") + len("Action:")
        action_input_idx = output_text.rindex("Action Input:")
        
        # 提取出工具名称，比如 " get_secret_info\n"
        action_name = output_text[action_idx:action_input_idx].strip()
        # 提取出参数，比如 " Rick"
        action_input = output_text[action_input_idx + len("Action Input:"):].strip()
        
        print(f"💡 [框架解析成功] LLM 想要调用工具: [{action_name}] 参数是: [{action_input}]")
        
        # 4. 执行本地 Python 函数!
        if action_name in tool_map:
            tool_func = tool_map[action_name]
            observation_result = tool_func(action_input)
            print(f"🔧 [本地工具执行] 返回结果: {observation_result}")
        else:
            observation_result = f"工具 {action_name} 不存在"
            print(f"⚠️ [本地工具执行报错] {observation_result}")
            
        # 5. 把 Python 运行结果强制写进记事本，让 LLM 下一轮能看见
        chat_history += f"Observation: {observation_result}\n"
        
    except Exception as e:
        print(f"❌ 解析 LLM 输出失败: {e}。这意味着 LLM 没有遵守 Prompt 格式。强制终止。")
        break
