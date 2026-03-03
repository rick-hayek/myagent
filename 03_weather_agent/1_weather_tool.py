import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
print(f"open weather api key: {openweather_api_key}")

@tool
def get_weather(city: str) -> str:
    """这是一个天气查询工具。当你需要查询指定城市的天气时，请调用此工具。
    Args:
        city: 想要查询的天气城市名称。⚠️警告：无论用户是用中文还是其他语言提问，你都必须在此处将城市名称翻译为纯英文全拼（如用户问"上海"，这里必须传入"Shanghai"），否则第三方天气接口将返回 404 错误。
    """
    # 坑点 1：缺少 API Key 的优雅处理
    # 不要让程序直接抛出异常崩溃退出（比如直接 raise ValueError），
    # 最佳实践是返回一段友好的字符串报错给大模型，让大模型知道发生了什么并转化为自然语言转告人类。
    if not openweather_api_key:
        return "【系统报错返回】由于系统后台没有配置 OPENWEATHERMAP_API_KEY，天气查询工具当前不可用。请向用户致歉并告知需联系管理员配置 API Key。"
    
    # 构建 OpenWeatherMap 请求 URL (units=metric 表示摄氏度, lang=zh_cn 表示中文描述)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}&units=metric&lang=zh_cn"

    try:
        # 坑点 2：调用外部网络请求时，务必加 timeout 网络超时时间！
        # 如果不加，一旦遇到网络波动，整个 Agent 的死循环逻辑会无限卡死在这里挂起。
        response = requests.get(url, timeout=5) # timeout: seconds
        
        # 坑点 3：处理 HTTP 报错状态码 (比如城市名输错导致 404)
        # 最坏的做法是任由底层报错炸毁程序；最好的做法是将报错原因描述成文本，喂回给大模型。
        if response.status_code == 404:
            return f"【查询失败】[code: {response.status_code}]找不到气象库中名为 '{city}' 的城市。你应当让用户核对城市名称是否拼写正确，或尝试输入英文全拼。"
        elif response.status_code == 401:
            print(f"response: {response.text}")
            return f"【查询失败】[code: {response.status_code}] API Key 鉴权失败。请致歉并告知用户后台的 API Key 可能已失效。"
        
        response.raise_for_status() # 拦截其他未知错误抛出到下面的 except
        
        # 4. 正常解析并拼装精简版数据给大模型看
        data = response.json()
        print(f"weather api response data: {data}")
        temp = data["main"]["temp"]
        weather_desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        
        return f"{city} 的当前实况天气：{weather_desc}，实时气温：{temp}℃，湿度：{humidity}%。你可以基于这些数据自由组合发散回复。"
        
    except requests.exceptions.Timeout:
        return "【网络超时返回】请求第三方天气服务耗时过长。请告诉用户网络开小差了，稍稍后再试。"
    except Exception as e:
        return f"【未知系统错误】调用天气 API 接口时出现代码级崩溃，错误详情: {str(e)}。请向用户报告错误详情。"

# ================= 简单的本地单测 =================
# 在不启动大模型的情况下，我们作为"人类"可以直接验证工具函数的健壮性
if __name__ == "__main__":
    print("[本地直接传参测试 - 查询真实存在的上海（英文请求）]:")
    print("工具返回内容 =>", get_weather.invoke({"city": "Shanghai"}))
    
    print("[本地直接传参测试 - 查询真实存在的上海（中文请求）]:")
    print("工具返回内容 =>", get_weather.invoke({"city": "上海"}))

    print("\n[本地直接传参测试 - 故意传入一个乱码不存在的城市触发 404]:")
    print("工具返回内容 =>", get_weather.invoke({"city": "Asdfghjkl"}))