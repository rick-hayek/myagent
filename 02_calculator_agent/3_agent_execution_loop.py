import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。
    Args:
        a: 第一个被乘数
        b: 第二个乘数
    """
    print(f"\n[🔧 本地工具 multiply 执行中... 检测到参数 a={a}, b={b}]") # 增加打印看看工具是不是真的在此执行
    return a * b

@tool
def add(a: int, b: int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    print(f"\n[🔧 本地工具 add 执行中... 检测到参数 a={a}, b={b}") # 增加打印看看工具是不是真的在此执行
    return a + b

tools = [multiply, add]
tools_by_name = {t.name: t for t in tools}

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

# 1. 模拟第一次提问：包含我们希望模型自己执行计算的问题
# 注意我们引入了消息列表(messages)，因为 Agent 需要记录聊天上下文
messages = [
    ("user", "先计算123乘以456等于多少？算出结果后，把它加上25。")
]

print(f"[用户原始问题]: {messages[0][1]}\n")


# 2. 第一次调用：大模型决定发号施令
print("[第一回合：大模型思考中...]")
response_message = llm_with_tools.invoke(messages)
print("第一回合思考返回: ", response_message)
messages.append(response_message) # Agent 记录了大模型的回复: 把大模型的指令存进聊天记录里

# 3. 检查大模型是不是派发了任务 (tool_calls)
if response_message.tool_calls: # 如果大模型提出了工具调用的指令
    print(f"[大模型决定不直接回答，而是分配了{len(response_message.tool_calls)}个本地工具执行任务]")
    # 4. 遍历大模型分配给我们的工具调用请求
    for tool_call in response_message.tool_calls:
        print(f"👉 决定调用工具: {tool_call['name']}, 参数准备: {tool_call['args']}")
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        # 从我们本地注册的工具字典里找到对应的原生 Python 函数
        selected_tool = tools_by_name[tool_name]

        # 5. 【执行器】真正的动手跑代码发生在这里！
        tool_result = selected_tool.invoke(tool_args)
        print(f"[本地代码执行完毕！计算结果直接得出了: {tool_result}]")

        # 6. 【最关键的一步】：必须把计算结果包装成 ToolMessage，喂回给大模型！
        # 这个动作相当于跟大模型汇报：“首长，工具我替你跑完了，结果是这个，请过目！”
        from langchain_core.messages import ToolMessage
        messages.append(ToolMessage(
            tool_call_id=tool_call['id'], # 我们要告诉它是针对它刚才发的哪条请求
            name=tool_name, # 我们要告诉它是针对哪个工具
            content=str(tool_result) # 把答案作为字符串返回给它看
        ))
        
    # 7. 第二次调用：大模型根据工具执行结果继续思考
    print("\n[第二回合：大模型查收刚才的计算结果报告，并总结最终发言...]")
    final_response = llm_with_tools.invoke(messages)
    print("\n🎉【大模型的最终人类自然语言回复】:\n", final_response.content)

else:
    print("[大模型觉得这个问题不需要使用工具，直接输出了自然语言回答]:", response_message.content)
    
        