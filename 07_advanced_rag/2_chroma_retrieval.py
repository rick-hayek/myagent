import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate

set_debug(True)
load_dotenv()

# ==========================================
# 步骤 1: 瞬间恢复记忆 (直接挂载本地 ChromaDB)
# ==========================================
print("🔄 正在从硬盘瞬间唤醒企业级知识库...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
# 直接挂载之前入库的文件夹，速度仅需 0.1 秒！不需要再切文件，不需要再算向量！
vectorstore = Chroma(
    persist_directory="./my_local_vectordb", 
    embedding_function=embeddings
)

# ==========================================
# 步骤 2: 解决 k=2 的惨剧 (引入 MMR 最大边缘相关性)
# ==========================================
# 这一次，我们把 k 调到 5，并且引入 MMR 策略。
# MMR: 先粗筛 10 个（fetch_k），再从中精选 5 个（k）互相最不重复的。
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

# 抛出一个拥有四个连续大坑的问题
# 1. Rick用大葱换了什么交通工具？
# 2. 它的开机密码是多少？
# 3. 外星人的结局是什么？
# 4. 米雪和Rick什么关系？ # 这个问题无法从参考资料中获取，看看模型怎么处理
query = "Rick用大葱换了什么交通工具？它的开机密码是多少？外星人的结局是什么？米雪和Rick什么关系？"

print(f"\\n🔍 收到超级复杂问题: {query}")
retrieved_docs = retriever.invoke(query)

print("📄 检索到的相关文档：")
for doc in retrieved_docs:
    print(f"文档内容：{doc.page_content}\n")

print(f"\\n✅ 找到了 {len(retrieved_docs)} 块提供不同信息维度的碎片材料！")
context_text = "\\n\\n".join([doc.page_content for doc in retrieved_docs])

# ==========================================
# 步骤 3: 最终推理 (Generation)
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt_template = """
你是一个严谨的资料分析助手。
请你严格根据下方的【参考资料】来回答用户的问题。
【参考资料开始】
{context}
【参考资料结束】
用户问题：{question}
"""

prompt = PromptTemplate.from_template(prompt_template)

final_chain = prompt | llm
response = final_chain.invoke({
    "context": context_text,
    "question": query
})

print("\n" + "="*50)
print("🎉 最终答案：")
print(response.content)
print("="*50)