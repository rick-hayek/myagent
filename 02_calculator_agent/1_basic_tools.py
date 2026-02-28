from langchain_core.tools import tool

# @tool 装饰器是 LangChain 里极其核心的魔法
# 它会把一个普通的 Python 函数，打包成大模型能够理解的 Tool (工具)
# ⚠️警告：函数下方的 """docstring(文档字符串)""" 绝不仅仅是人类看的注释！
# 大模型就是靠阅读这段说明，来决定它什么时候应该调用这个工具的！
@tool
def multiply(a: int, b: int) -> int:
    """这是一个乘法器。当你需要计算两个数字相乘时，请调用此工具。
    Args:
        a: 第一个被乘数
        b: 第二个乘数
    """
    return a * b

@tool
def add(a: int, b:int) -> int:
    """这是一个加法器。请用来计算两个数字相加的和。"""
    return a + b

# 我们可以把定义好的工具放进一个数组里，随时准备丢给大模型
tools = [multiply, add]

# 我们可以验证一下，它现在已经不再是普通的函数了
print("工具在系统中的名字：", multiply.name)
print("大模型看到的工具描述说明书：", multiply.description)
print("大模型看到的数据结构：", multiply.args_schema.model_json_schema())
print("工具的参数模式：", multiply.args)

# 我们甚至可以直接在本地，不通过大模型，直接手动测试调用一下这个封装好的工具对象
result = multiply.invoke({"a": 3, "b": 4})
print("\n[本地测试向工具传入参数调用]: 3 * 4 =", result)