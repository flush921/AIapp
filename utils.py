from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import base64
import os
from io import BytesIO
from langchain.memory import ConversationBufferWindowMemory

#sk-c4L1f9e7fb0ead5e5a76ce561bbe98957b083d34a55dg74i
def get_chat_response_dp(prompt, memory, openai_api_key, modelname):
    if modelname == "deepseek-V3":
        model = ChatOpenAI(
            model = 'deepseek-chat',
            openai_api_key = openai_api_key,
            openai_api_base = 'https://api.deepseek.com',
        )
        chain = ConversationChain(llm = model,memory = memory)
        response = chain.invoke({"input":prompt})
        return response["response"]
    elif modelname == "deepseek-R1":
        return get_chat_response_dpr1(prompt, memory, openai_api_key)


def get_chat_response_dpr1(prompt, memory, openai_api_key):
    model = ChatOpenAI(
        model = 'deepseek-reasoner',
        openai_api_key = openai_api_key,
        openai_api_base = 'https://api.deepseek.com',
    )
    chain = ConversationChain(llm = model,memory = memory)
    response = chain.invoke({"input":prompt})
    return response["response"]


# 定义一个常量PLATFORMS，仅包含OpenAI平台
PLATFORMS = ["OpenAI"]

# 定义一个函数get_llm_models，用于获取OpenAI平台的LLM模型列表
def get_llm_models(platform_type: str= "OpenAI", base_url: str = "https://api.gptsapi.net", api_key: str = ""):
    if platform_type == "OpenAI":
        # 返回OpenAI预定义的模型列表
        return [
            'gpt-4o',
            'gpt-3.5-turbo'
        ]
    return []  # 理论上不会执行到这里

# 定义一个函数get_embedding_models，用于获取OpenAI平台的Embedding模型列表
def get_embedding_models(platform_type: str= "OpenAI", base_url: str = "", api_key: str = "EMPTY"):
    # 仅保留OpenAI平台的逻辑（此处简化返回示例模型）
    if platform_type == "OpenAI":
        return ["text-embedding-ada-002"]
    return []

# 定义一个函数get_chatllm，用于获取OpenAI平台的聊天模型
def get_chatllm(
        platform_type: str= "OpenAI",
        model: str = "",
        base_url: str = "https://api.gptsapi.net/v1",
        api_key: str = "sk-c4L1f9e7fb0ead5e5a76ce561bbe98957b083d34a55dg74i",
        temperature: float = 0.1
):
    # 仅保留OpenAI平台的逻辑
    if platform_type == "OpenAI":
        # 如果没有提供base_url，使用默认的OpenAI地址
        if not base_url:
            base_url = "https://api.gptsapi.net/v1"

        # 返回ChatOpenAI对象
        return ChatOpenAI(
            temperature=temperature,
            model_name=model,
            streaming=True,
            base_url=base_url,
            api_key=api_key,
        )
    return None  # 理论上不会执行到这里

# 定义一个函数get_kb_names，用于获取知识库的名称列表
def get_kb_names():
    # 获取当前文件所在目录下的kb目录路径
    kb_root = os.path.join(os.path.dirname(__file__), "kb")
    # 如果kb目录不存在，则创建该目录
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    # 获取kb目录下的所有子目录名称，并返回这些名称
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]
    return kb_names

# 定义一个函数get_embedding_model，用于获取OpenAI平台的Embedding模型
def get_embedding_model(
        platform_type: str= "OpenAI",
        model: str = "text-embedding-ada-002",
        base_url: str = "https://api.gptsapi.net/v1",
        api_key: str = "sk-c4L1f9e7fb0ead5e5a76ce561bbe98957b083d34a55dg74i",
):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)

# 定义一个函数get_img_base64，用于获取图片的base64编码
def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    # 获取当前文件所在目录下的img目录中的图片路径
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    # 读取图片
    with open(image_path, "rb") as f:
        # 将图片数据读入内存
        buffer = BytesIO(f.read())
        # 对图片数据进行base64编码，并解码为字符串
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # 返回base64编码的图片数据，格式为data:image/png;base64,编码数据
    return f"data:image/png;base64,{base_str}"


# api测试
#memory = ConversationBufferWindowMemory(k = 5,return_messages=True)
#print(get_chat_response("笛卡尔提出了哪些有名的定律？",memory,"sk-64dd601abf7a44368a48c293de9f536f"))
#print(get_chat_response("我的上一个问题是什么？要求一字不差",memory,"sk-64dd601abf7a44368a48c293de9f536f"))

