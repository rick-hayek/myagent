import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ==========================================
# Indexing：极其消耗算力，这辈子只需运行一次
# ==========================================
# 1. 提取数据
print("🔄 正在装载绝密文件...")
loader = TextLoader("06_simple_rag/my_secret_data.txt", encoding="utf-8")
docs = loader.load()

# 2. 注入中文灵魂，优雅切分
print("✂️ 正在使用中文神级切刀分割文件...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=10,
    separators=["\\n\\n", "\\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"文档被切成了 {len(chunks)} 块")
for chunk in chunks:
    print(f"文档碎片：{chunk.page_content}\n")

print("🧠 正在调用 Embedding 模型将文字转化为高维数字向量，并存入本地磁盘向量库 Chroma...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# ==========================================
# 史上最强改动：持久化到本地 ChromaDB！
# ==========================================
print("💾 正在将向量写入本地实体硬盘 (ChromaDB) ...")
# 这一步会自动在当前目录下生成一个名为 "my_local_vectordb" 的文件夹
# 哪怕电脑断电，这些包含你企业机密的数字向量也将与世长存！
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./my_local_vectordb" # 【关键魔法】：指定硬盘存放路径
)

print("✅ Indexing 入库完成！请去 myagent 根目录下膜拜你人生中第一个实体向量数据库文件夹吧！")