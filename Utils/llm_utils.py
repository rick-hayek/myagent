from langchain_core.messages import AIMessage

def response_to_str(response: AIMessage) -> str:
    content = response.content
    if isinstance(content, list):
        text = content[0]["text"]
    else:
        text = content

    return text