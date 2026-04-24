import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

load_dotenv()

qwen_model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
deepseekchat_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

text = "给生产纸杯的公司取一个名字，直接输出名字"
messages = [HumanMessage(content=text)]
if __name__ == "__main__":
    print("千问结果")
    print(qwen_model.invoke(messages).content)
    print("deepseek结果")
    print(deepseekchat_model.invoke(messages).content)