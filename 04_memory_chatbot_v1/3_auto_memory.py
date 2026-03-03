import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# 引入 LangChain 的内存系统基石
from langchain_core.chat_history import InMemoryChatMessageHistory
# 引入自动包裹记忆的"胶囊"
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.globals import set_debug

set_debug(True)

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1. 准备一个类似仓库的字典，用来存放不同人的聊天记录
# 真实产品里这里可能是连着 Redis 或者 MySQL，今天我们先用内存字典
store = {}

# 2. 这是一个获取历史记录的函数，当接收到 session_id 时，去仓库里找他的专属小本本
def get_session_memory(session_id: str):
    # 如果这个 session_id 不存在，就给他开一个新的空白记忆本
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. 最激动人心的一步：把 "没记忆的 llm" 封装进 "记忆胶囊"
# 它会在每次调用前自动去 store 里取历史，调用后自动把结果存进去
llm_with_memory = RunnableWithMessageHistory(
    llm, # 被包裹的裸模型
    get_session_memory # 获取记忆本的函数规则
)

print("🤖 拥有【全自动短期记忆】的客服已启动 (输入 'quit' 退出)")
print("-" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    # 我们不需要在这里手敲任何 .append() 了！
    # 只需要在 invoke 的第二个参数 config 里告诉它，我是谁（session_id）。
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )

    print(f"AI: {response.content}\n")
