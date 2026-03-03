import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 召唤没有任何记忆外挂、纯粹的裸模型
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("🤖 聊天机器人已启动 (输入 'quit' 退出)")
print("-" * 50)

# 我们写一个死循环，模拟类似微信对话框的人机一问一答的交互
while True:
    user_input = input("\n你: ")
    if user_input.lower() == "quit":
        break
    
    # 我们把人类每次敲的新问题，直接丢给裸模型
    response = llm.invoke(user_input)
    print(f"AI: {response.content}\n")