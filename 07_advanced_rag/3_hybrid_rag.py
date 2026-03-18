import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

# ==========================================
# 准备阶段：把文档加载出来喂给它
# ==========================================
print("🔄 准备基础数据...")
loader = TextLoader("06_simple_rag/my_secret_data.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, chunk_overlap=10, separators=["\\n\\n", "\\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_documents(docs)

# ==========================================
# 步骤 1: 构建双核检索引擎
# ==========================================
print("⚙️ 引擎 A：正在构建传统的 BM25 关键词检索器...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 2

print("⚙️ 引擎 B：正在挂载 Chroma 语义向量检索器...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
chroma_vectorstore = Chroma(persist_directory="./my_local_vectordb", embedding_function=embeddings)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

# ==========================================
# 终极融合：自行实现组合检索器 (RRF 算法)
# ==========================================
print("🚀 终极融合：正在使用 RRF 算法组装双引擎...")
def custom_ensemble_retrieve(query, retrievers, weights, c=60):
    rrf_score = {}
    
    for retriever, weight in zip(retrievers, weights):
        docs = retriever.invoke(query)
        for rank, doc in enumerate(docs):
            doc_content = doc.page_content
            score = weight / (rank + c + 1) # 加入极小的平滑处理防止除0
            
            if doc_content not in rrf_score:
                rrf_score[doc_content] = {"doc": doc, "score": 0.0}
            rrf_score[doc_content]["score"] += score
            
    # 按综合得分从高到低排序，取前 3 名
    sorted_docs = sorted(rrf_score.items(), key=lambda x: x[1]["score"], reverse=True)
    return [item[1]["doc"] for item in sorted_docs[:3]]

# ==========================================
# 步骤 2: 发出查询请求
# ==========================================
query = "小明在哪颗星球上遇害的？"
print(f"\\n🔍 混合提问: {query}")

# 调用我们的手写合并器
retrieved_docs = custom_ensemble_retrieve(
    query, 
    retrievers=[bm25_retriever, chroma_retriever], 
    weights=[0.3, 0.7]
)

print("📄 检索到的相关文档：")
for doc in retrieved_docs:
    print(f"文档内容：{doc.page_content}\\n")

print(f"\\n✅ 混合检索完毕！双引擎联合推荐了 {len(retrieved_docs)} 块无敌碎片！")
context_text = "\\n\\n".join([doc.page_content for doc in retrieved_docs])

# ==========================================
# 步骤 3: 最终推理 (Generation)
# ==========================================
print("\\n🤖 正在提交给大模型作答...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
rag_prompt = PromptTemplate.from_template("""
请严格根据参考资料回答。
【参考资料】
{context}
用户问题：{question}
""")

final_chain = rag_prompt | llm
response = final_chain.invoke({
    "context": context_text,
    "question": query
})

print("\\n" + "="*50)
print("🎉 最终答案：")
print(response.content)
print("="*50)