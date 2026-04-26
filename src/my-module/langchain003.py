import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

qwen_model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

deepseek_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

template = "你是一个能将{input_language}翻译成{output_language}的助手"
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human",human_template)
])

qwen_chain = chat_prompt | qwen_model | StrOutputParser()
deepseek_chain = chat_prompt | deepseek_model | StrOutputParser()

if __name__ == "__main__":
    input_data = {
        "input_language": "汉语",
        "output_language": "英语",
        "text": "我爱编程"
    }

    print(f"--- 正在翻译：'{input_data['text']}' ---\n")
    print("千问回答"）
    qwen_res = qwen_chain.invoke(input_data)
    print(qwen_res)

    print("deepseek回答")
    deepseek_res = deepseek_chain.invoke(input_data)
    print(deepseek_res)