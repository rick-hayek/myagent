# AI Agent 实战开发备忘录 (Step-by-Step Tutorial)

此文档用于记录我们在开发过程中的每一步实际操作，作为详尽的动手指南。

## 准备工作：环境初始化

我们在正式开始编码前，首先需要搭建一个隔离的 Python 环境并安装必需的库。

### 步骤 1：配置虚拟环境与核心依赖

**目标**：创建一个独立干净的开发环境，避免与 Mac 系统的全局 Python 库产生冲突，并按需安装第一阶段需要的库。

**执行命令**：
在项目根目录 `/Users/rick/src/myagent` 下的终端中执行：

```bash
# 1. 创建名为 venv 的虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境 (你会看到命令行前缀多出了 (venv))
source venv/bin/activate

# 3. 安装 LangChain 核心、Google 官方纯 SDK、LangChain Google 集成、以及环境变量读取库
pip install -U langchain google-genai langchain-google-genai python-dotenv
```

### 步骤 2：使用纯 SDK 请求大模型

**目标**：不依赖高级框架，直接使用 Google 官方自带的 Python SDK `google-genai` 进行底层调用。这有助于我们在未来使用 LangChain 遇到报错时，排查是否是底层接口或 Key 的问题。

**代码实现** (`01_basic_api/1_ai_sdk_call.py`)：

```python
import os
from dotenv import load_dotenv
from google import genai

# 1. 自动寻找并加载环境中的 .env 文件
load_dotenv()

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# 2. 实例化官方无头 SDK 客户端
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print(f"User input: Explain how AI works in a few words")

# 3. 调用底层 API 生成内容
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

# 4. 打印纯文本结果
print(f"AI response: {response.text}")
```

**运行命令**：
```bash
python3 01_basic_api/1_ai_sdk_call.py
```

### 步骤 3：使用 LangChain 调用大模型

**目标**：引入 LangChain 框架，体验它是如何将各种底层的大模型接口（如 Google, OpenAI）抽象包装成统一格式的。

**代码实现** (`01_basic_api/2_langchain_basic.py`)：

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 实例化 LangChain 的大模型包装类
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
print("通过LangChain调用Google Gemini API")
print("input: Explain how AI works in a few words")

# 调用 invoke 方法向模型提出问题
response = llm.invoke("Explain how AI works in a few words")

# 解析并打印回答内容。注意 response 是一个 AIMessage 对象。
print("\n [LangChain 返回的完整对象类型]:", type(response))
print("\n [AI 生成的文本内容]:", response.content)
```

**运行命令**：
```bash
python3 01_basic_api/2_langchain_basic.py
```
**核心观察**：无论底层是哪个模型，LangChain 返回的始终是标准化的 `AIMessage` 对象。

### 步骤 4：引入提示词模板 (Prompt Templates)

**目标**：学习使用 `ChatPromptTemplate`，将用户角色（System/User）和动态变量（{user_input}）结合起来，体会大模型应用中不可或缺的 Prompt 工程化。

**代码实现** (`01_basic_api/3_prompt_template.py`)：
```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. 创建一个 LLM 实例
llm = ChatGoogleGenerativeAI(model="gemini-3-flash")

# 2. 创建一个带有 system 设定和 user 动态变量的聊天提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的英语翻译官...只输出翻译结果，不要带任何多余解释。"),
    ("user", "给我翻译这句话：{user_input}"),
])

# 3. 利用 LCEL 管道符将模板和大模型组合拼接
chain = prompt | llm

user_text = input("请输入你想翻译的中文互联网黑话: ")


# 4. 执行这套组合逻辑，传入变量字典
response = chain.invoke({"user_input": user_text})

print("\n[最终翻译结果]:")
print(response.content)
```

**核心观察**：
`chain = prompt | llm` 是 LangChain 最灵魂的用法，称为 **LCEL** (LangChain Expression Language)。它代表将左侧的处理结果“流式”传递给右侧。

### 步骤 5：引入输出解析器 (Output Parsers)

**目标**：掌握如何使用 `StrOutputParser` 获取纯文本，以及使用 `JsonOutputParser` 结合 Pydantic 获取结构化的 Python 字典数据，为后续 Agent 的稳定运行打下基础。

**代码实现** (`01_basic_api/4_output_parser.py`)：
```python
import pydantic
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("=============== 测试一：字符串解析器 ===============")
prompt1 = ChatPromptTemplate.from_template("请用一句话夸奖这款产品: {product}")
chain1 = prompt1 | llm | StrOutputParser()

input_product = input("请输入产品名称: ")
text_response = chain1.invoke({"product": input_product})
print("[纯文本结果]:", text_response)

print("\n=============== 测试二：JSON 解析器 ===============")
class Joke(BaseModel):
    setup: str = Field(description="笑话的铺垫部分")
    punchline: str = Field(description="笑话的包袱/笑点部分")

json_parser = JsonOutputParser(pydantic_model=Joke)
prompt2 = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话。\n{format_instructions}")
chain2 = prompt2 | llm | json_parser

json_response = chain2.invoke({
    "topic": "程序员",
    "format_instructions": json_parser.get_format_instructions()
})
print("[字典解析结果]:", json_response)


print("\n=============== 测试三：现代结构化输出 ===============")
# 1. 现代解法：在 API 底层强制 LLM 绑定 Pydantic 数据结构 (彻底取代 JsonOutputParser)
# 只要模型不支持这个结构，它就会在生成前直接抛错，避免"幻觉"格式
structured_llm = llm.with_structured_output(Joke)

# 2. 提示词变得极度纯净，无需再硬塞 format_instructions 占位符
prompt3 = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话。")

# 3. 组装新管道: Prompt -> 强约束的 LLM
chain3 = prompt3 | structured_llm

# 4. 执行生成
structured_response = chain3.invoke({"topic": "产品经理"})

# 返回的直接是被验证过 100% 符合 Joke 结构的 Pydantic 类的实例对象
print("[结构化输出强约束返回的数据]:", structured_response)
print("[结构化输出数据类型]:", type(structured_response))
# 像操作普通对象一样，直接用对象点属性名的方式取值
print("[坚固提取 setup]:", structured_response.setup)
print("[坚固提取 punchline]:", structured_response.punchline)
```

**核心观察 1（传统解析）**：`JsonOutputParser` 会把 Pydantic 类的结构说明转义成一段很长的提示词喂给大模型，拿到长文本后会自动尝试反序列化成 Python 原生的字典。但这种基于 Prompt 的方式在弱模型下可能会遭到“指令无视”导致返回 None。

**核心观察 2（现代约束）**：`.with_structured_output()` 是当今 Agent 开发中最先进、成功率最高的结构化提取方式。它是 API 底层特性，直接返回 Pydantic 对象（通过 `.` 访问属性），从而彻底消除了传统提示词方案的不稳定性。

### 步骤 6：体验打字机效果的流式输出 (Streaming)

**目标**：掌握大模型生成长文本时的最佳用户体验实践——流式输出，让客户端实现像 ChatGPT 一样的打字机打印效果。

**代码实现** (`01_basic_api/5_streaming.py`)：
```python
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_template("请帮我写一首关于{topic}的长诗，至少包含四段。")
chain = prompt | llm | StrOutputParser()

topic_input = input("请输入你想写诗的主题: ")
print("\n[AI 正在努力作诗中，请欣赏终端里的打字机效果]...\n")

# 【核心差异】不使用 invoke()，改用 stream() 方法获取流式迭代器
for chunk in chain.stream({"topic": topic_input}):
    # print 默认会在每个字符后加换行，我们改成加空字符串，并强制立刻刷新输出流(flush=True)
    print(chunk, end="", flush=True)

print("\n\n[诗歌创作完毕！]")
```

**核心观察**：
`invoke()` 是一次性返回完整结果（会有很长的白屏等待期），而 `stream()` 是一有新内容就立刻返回。配合 `print` 的 `flush=True` 强制缓冲刷新，就能在终端完美复刻商业级 AI 应用的流式体验。

---

### Phase 1 核心知识点总结：解析器 vs 结构化输出的底层机制区别

在体验了大模型提取结构化数据（JSON）的过程中，我们使用了两种截然不同的方案。理解它们的差异是掌握现代 Agent 稳定性的关键：

#### 1. `JsonOutputParser` 的底层机制
它做的事情其实很"笨"，分两步：
1. **指令拼接**：调用 `json_parser.get_format_instructions()` 会自动将你定义的 Pydantic 数据类转换成一段纯文本的严格格式说明（比如要求包含某些特定的字段名和类型）。这段文字被硬塞进 Prompt（提示词）末尾发给模型，本质上就是在"求"模型按格式输出。
2. **事后验证**：拿到模型返回的普通自然语言文本后，Parser 尝试用 `json.loads()` 等手段事后解析它。

**问题根源**：模型返回的是自然语言文本，Parser 只是在事后尝试解析。模型如果是强大的（如 GPT-4），可能愿意乖乖按格式来；如果模型较弱或者被前置任务绕晕了，就会忽略末尾的格式要求乱输出文本。一旦格式不对，Parser 就会报错或提取出 `None`，因为它对模型生成过程**完全没有控制权**。

#### 2. `with_structured_output` 的底层机制
这是现代 Agent 开发的标准解法，它走的是完全不同的路径，利用的是模型的 **Function Calling（函数调用）/ 强结构化特性**。
具体来说：
1. 你定义的 Pydantic 结构会被直接转换成严谨的 JSON Schema。
2. 这个 Schema 是通过 **API 参数级别**（而不是丢进 Prompt 文本里）硬性传给大模型的。
3. 模型在逐字生成（Token 生成机制）的底层阶段就被强行约束：它只能输出符合这个 Schema 规则的 Token，一旦预测偏离就会被 API 底层拦截重绘。
4. 最终返回给你的甚至都不再是纯文本，而是已经被 LangChain 直接反序列化好的原生 **Pydantic 对象实例**。

**关键区别**：约束发生在**生成过程中**，而不是生成之后。模型根本没有任何机会"乱输出"废话。

**一个绝佳的类比**：
- `JsonOutputParser` 像是你告诉一个人：*"请在这张白纸上画一个表格然后把答案填进去"*。你拿到白纸后，自己去猜画成什么样的表格、以及数据怎么填。
- `with_structured_output` 像是你直接递给他一张**已经印好死框框的 Excel 填空表**，他只能且必须在指定的格子里写内容，想写到格子外面都不可能。

---

> **恭喜！你已通关 Phase 1：大模型核心三剑客（API、Prompt、OutputParser）。**
> 下一步，我们将进入真正让大模型变成 Agent (智能体) 的核心阶段：**Tool Calling (工具调用/函数调用)**。
