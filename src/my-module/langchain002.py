import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate

load_dotenv()

qwen_model = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
deepseekchat_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

prompt = PromptTemplate.from_template("给生产{product}的公司取一个名字，直接输出名字")
qwen_chain = prompt | qwen_model
deepseek_chain = prompt | deepseekchat_model

if __name__ == "__main__":
    product_name = "冯承坤"
    print("千问结果")
    print(qwen_chain.invoke({"product": product_name}).content)
    print("deepseek结果")
    print(deepseek_chain.invoke({"product": product_name}).content)