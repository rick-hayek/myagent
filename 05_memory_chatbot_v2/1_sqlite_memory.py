import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
# 【关键改变 1】：引入连接硬盘数据库的全新记忆力类
from langchain_community.chat_message_histories import SQLChatMessageHistory

# 不要忘了我们刚学的"第一性原理"上帝视角
from langchain_core.globals import set_debug
set_debug(True)

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 【关键改变 2】：我们不再需要 store = {} 这个脆弱的内存字典了！
def gete_session_memory(session_id: str):
    # 【关键改变 3】：直接返回一个连接到本地 SQLite 数据库的历史对象
    # 如果对应 session_id 的记录不存在，它会自动建表；如果存在，它会自动把过去的所有对话全部捞出来！
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db" # 数据库存放在本地当前目录的这个文件里
    )

llm_with_memory = RunnableWithMessageHistory(
    llm,
    get_session_history=gete_session_memory
)

print("💾 拥有【数据库长期记忆】的客服已启动 (输入 'quit' 退出)")
print("-" * 50)

# 我们写一个死循环，模拟类似微信对话框的人机一问一答的交互
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
        
    response = llm_with_memory.invoke(
        user_input, 
        config={"configurable": {"session_id": "user_rick"}}
    )
    print(f"AI: {response.content}\n")
