import pydantic
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# 引入 LangChain 内置的响应解析器组件
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("=============== 测试一：字符串解析器 (StrOutputParser)===============")

prompt1 = ChatPromptTemplate.from_template("请用一句话夸奖这款产品: {product}")

# 组装管道：Prompt -> LLM -> String Output Parser
# StrOutputParser 的作用是自动把 LLM 返回的厚重的 AIMessage 对象剥开，只把最里面的纯文本提取出来返回给你
chain1 = prompt1 | llm | StrOutputParser()

#input_product = input("请输入产品名称: ")
input_product = "索尼电视"
# 此时 text_response 不再是带有 content 属性的对象，而是原汁原味的字符串
text_response = chain1.invoke({"product": input_product})
print("[字符串解析器返回的纯文本结果]:", text_response)

print("\n=============== 测试二：JSON 解析器 (JsonOutputParser)===============")

# 我们先用 Pydantic 定义我们预期大模型返回的严格 JSON 数据结构
class Joke(BaseModel):
    setup: str = Field(description="笑话的铺垫部分")
    punchline: str = Field(description="笑话的包袱/笑点部分")

# 根据上面定义的数据结构，实例化一个对应的 JSON 解析器
json_parser = JsonOutputParser(pydantic_model=Joke)

# 注意在 Prompt 中，我们需要向 AI 注入解析器自动生成的“格式说明指令”
prompt2 = ChatPromptTemplate.from_template(
    "讲一个关于{topic}的冷笑话。\n{format_instructions}"
)
# 组装管道：Prompt -> LLM -> JSON Output Parser
chain2 = prompt2 | llm | json_parser


# 执行时，把 format_instructions 变量传进去
json_response = chain2.invoke({
    "topic": "程序员",
    "format_instructions": json_parser.get_format_instructions()
})
print("[JSON 解析器返回的结构化数据]:", json_response)
print("[JSON 解析器返回的数据类型]:", type(json_response))
print("[提取特定字段 - setup]:", json_response.get('setup'))
print("[提取特定字段 - punchline]:", json_response.get('punchline'))


print("\n=============== 测试三：现代结构化输出 (with_structured_output)===============")
# 1. 直接让 LLM 绑定 Pydantic 数据结构！(不再需要 JsonOutputParser 了)
# 这是 API 底层的强约束，只要模型不支持这个结构它就会在生成前直接抛错，而不是乱生成。
structured_llm = llm.with_structured_output(Joke)
# 2. 提示词变得极度干净，不再需要 format_instructions 这个占位符
prompt3 = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话。")
# 3. 组装新管道: Prompt -> 强约束的 LLM
chain3 = prompt3 | structured_llm
# 4. 轻松执行
structured_response = chain3.invoke({"topic": "产品经理"})
# 返回的直接就一定是被验证过100%符合 Joke 结构的字典！
print("[结构化输出强约束返回的数据]:", structured_response)
print("[结构化输出数据类型]:", type(structured_response))
print("[坚固提取 setup]:", structured_response.setup)
print("[坚固提取 punchline]:", structured_response.punchline)