import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import trim_messages # LangChain 官方修剪器
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 【魔法时刻】：声明一个记忆修剪器 (Trimmer)
# 它的规则是：倒着数，不管前面有多少条历史记录，我只要最后 2 个 token 左右的内容（或者简单的说，只保留最近的两句话）
# 这里为了你能秒看效果，我故意把允许通过的长度 max_tokens 设置得非常激进（比如只允许活下来 40 个 token，折合大概一两句话）
trimmer = trim_messages(
    max_tokens=40,   # 非常短！测试专用
    strategy="last", # 从后往前数，保留最新的
    token_counter=llm, # 把计算每个字占多少 token 的工作交给这门模型自己
    include_system=True, # 如果有 System Prompt，绝对不能被裁掉
)

# 我们用 Linux 管道的思想：
# 把历史记录输送给 trimmer 修剪 -> 把修剪后剩下的精华输送给大模型 llm
chain_with_trimming = trimmer | llm

def get_session_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db" 
    )
    
# 封装时，传给它的不再是裸 llm，而是带有修剪器的 chain_with_trimming
llm_with_memory = RunnableWithMessageHistory(
    chain_with_trimming,
    get_session_history=get_session_memory,
)

print("✂️ 拥有【数据库记忆 + 自动修剪截断术】的客服已启动 (输入 'quit' 退出)")
print("-" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")