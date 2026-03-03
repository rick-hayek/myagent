import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("🤖 聊天机器人已启动 (输入 'quit' 退出)")
print("-" * 50)

# 1. 核心机制：在客户端维护一个列表，专门用来存储聊天历史
chat_history = []

# 我们写一个死循环，模拟类似微信对话框的人机一问一答的交互
while True:
    user_input = input("\n你: ")
    if user_input.lower() == "quit":
        break
    
    # 2. 用户说的话，立刻追加到历史记录里
    #chat_history.append({"role": "user", "content": user_input})
    #print(f"Human Message: {HumanMessage(content=user_input)}")
    chat_history.append(HumanMessage(content=user_input))
    
    # 3. 关键动作：我们不再是单纯发送 user_input，
    # 而是把累积了所有上下文的完整的 chat_history 列表，一股脑发送给模型！
    response = llm.invoke(chat_history)

    # 4. 把模型的回应，也追加到历史记录里，形成闭环！
    #chat_history.append({"role": "assistant", "content": response.content})
    #print(f"AI Message: {AIMessage(content=response.content)}")
    chat_history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}\n")

    # 我们打印一下幕后的情况，让你看清真相：
    print(f"  [幕后揭秘：当前发送给模型的聊天记录条数：{len(chat_history)} 条]")