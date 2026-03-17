import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 【核心黑科技】：手动写一个记忆摘要拦截器！
def summarize_messages(messages):
    print(f"\n[摘要器被唤醒] 当前拦截到 {len(messages)} 条历史记录。")
    # 如果聊天记录还很短（比如只有3条以内），大可不必浪费算力去总结，直接全额放行
    if len(messages) <= 3:
        print("[摘要器] 长度安全，直接放行不摘要。")
        return messages
    
    print("[摘要器] 触发阈值！正在调用大模型将旧对话压缩成摘要...")
    # 把最近的两句话抽出来原样保留，保证当下的聊天体验不断层
    old_messages = messages[:-2]
    recent_messages = messages[-2:]
    
    summary_prompt = "请用一句简短的话总结以下对话的核心内容，以第三人称描述：\n"
    for msg in old_messages:
        summary_prompt += f"{msg.type}: {msg.content}\n"
        
    summary_response = llm.invoke(summary_prompt)
    print(f"[摘要器] 压缩完成！长篇大论已变为 -> {summary_response.content}")
    
    summary_msg = SystemMessage(content=f"【前情提要（记忆摘要）】：{summary_response.content}")
    return [summary_msg] + recent_messages

chain_with_summary = summarize_messages | llm

def get_session_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history_summary.db" 
    )

llm_with_memory = RunnableWithMessageHistory(
    chain_with_summary,
    get_session_history=get_session_memory
)

print("📝 拥有【数据库记忆 + 自动摘要压缩术】的客服已启动 (输入 'quit' 退出)")
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