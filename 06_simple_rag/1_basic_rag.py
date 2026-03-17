from h11._abnf import chunk_size
import os
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate

# 开启上帝视角，等下重点看最后的综合 Prompt 是怎么拼接成的！
set_debug(True)
load_dotenv()

# ==========================================
# 步奏 1: Indexing (数据装载、切分与向量化入库)
# ==========================================
print("🔄 正在装载绝密文件...")
loader = TextLoader("06_simple_rag/my_secret_data.txt", encoding="utf-8")
docs = loader.load()

print("✂️ 正在将文件切碎...")
# 哪怕文件很小，在真实工业中我们也要按段落切块防超载
# 【坑点提醒】：默认的切分符是基于英文习惯的 ["\n\n", "\n", " ", ""]
# 对于中文本，必须加上中文标点符号，否则大段的中文会被当成一个“超长英文单词”导致切分发疯！
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=10,
    separators=[
        "\n\n",
        "\n",
        "。",  # 中文句号结尾
        "！",
        "？",
        "，",  # 中文逗号
        " ",
        ""
    ]
)
chunks = text_splitter.split_documents(docs)
print(f"文档被切成了 {len(chunks)} 块")
for chunk in chunks:
    print(f"文档碎片：{chunk.page_content}\n")

print("🧠 正在调用 Embedding 模型将文字转化为高维数字向量，并存入本地内存向量库...")
# 注意：这里用的不是聊天的大模型，而是专门用来算距离的 Embedding 模型
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 一键生成拥有向量搜索能力的数据库
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# ==========================================
# 步骤 2: Retrieval (用户提问并进行向量相似度检索)
# ==========================================
query = "Rick用大葱换了什么交通工具？它的开机密码是多少？外星人的结局是什么？"

print(f"\n🔍 收到问题: {query}")
print("🎯 正在向量数据库中搜寻最相关的只言片语...")
# 指定把相似度最高的前 2 块碎片找出来
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke(query)

print(f"\n✅ 找到了 {len(retrieved_docs)} 块高度相关的碎片材料！")
# 我们把找出来的几块碎片，强行合并成一段大长文（也就是背景上下文 Context）
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# ==========================================
# 步骤 3: Generation (合并成巨型 Prompt 喂给聊天大模型)
# ==========================================
print("\n🤖 正在把碎片材料和用户问题打包，提交给聊天大模型作答...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# 这是一个极其经典的 RAG 终极 Prompt 模板
rag_prompt = PromptTemplate.from_template("""
你是一个严谨的资料分析助手。
请你严格根据下方的【参考资料】来回答用户的问题。
如果你在【参考资料】中找不到答案，请直接回答"根据现有资料我无法完整解答此问题"，绝不能编造。
【参考资料开始】
{context}
【参考资料结束】
用户问题：{question}
""")

# 把拼好的字符串发给LLM
final_chain = rag_prompt | llm
response = final_chain.invoke({
    "context": context_text,
    "question": query
})

print("\n" + "="*50)
print("🎉 最终答案：")
print(response.content)
print("="*50)
