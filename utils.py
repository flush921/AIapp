from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import base64
import os
from io import BytesIO
from langchain.memory import ConversationBufferWindowMemory
import json
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent






#openai   sk-c4L1f9e7fb0ead5e5a76ce561bbe98957b083d34a55dg74i
#deepseek sk-d2d7129c0e614170bb3dc84102eb5724
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

def dataframe_agent(openai_api_key, df, query):
    model = ChatOpenAI(model = 'deepseek-chat',
                       openai_api_base='https://api.deepseek.com',
                       openai_api_key=openai_api_key,
                       temperature=0
    )
    agent = create_pandas_dataframe_agent(llm=model,
                                          df=df,
                                          agent_executor_kwargs={"handle_parsing_errors": True},
                                          verbose=True,
                                          allow_dangerous_code=True
                                          )
    prompt = PROMPT_TEMPLATE + query
    response = agent.invoke({"input": prompt})
    response_dict = json.loads(response["output"])
    return response_dict


PROMPT_TEMPLATE = """
你是一位数据分析助手，负责处理用户关于数据集的查询。请严格遵循以下流程和格式要求完成任务：

一、任务处理流程规范
1. 思考阶段（未完成任务时）：
   - 先分析用户需求：判断是否需要调用工具（如计算数据、统计分析）。
   - 若需要调用工具，必须输出：
     Thought: [描述你的思考过程，例如："用户需要计算卧室平均数，需调用python_repl_ast工具计算df['bedrooms'].mean()"]
     Action: python_repl_ast
     Action Input: [具体执行的代码，例如："df['bedrooms'].mean()"]

2. 最终输出阶段（任务完成后）：
   - 当工具返回结果或无需工具即可回答时，必须停止工具调用，输出最终答案。
   - 最终答案需严格用以下格式包裹，且仅包含该格式内容：
     Final Answer: [你的最终答案，必须符合下方JSON格式要求]

二、最终答案JSON格式要求
根据用户需求类型，最终答案需按以下JSON格式返回（必须是可解析的标准JSON）：

1. 文字回答（如描述性结论）：
   {"answer": "<用自然语言描述的答案，字符串需用双引号包围>"}
   示例：{"answer": "数据集中卧室的平均数为2.97"}

2. 表格输出（如分类统计结果）：
   {"table": {"columns": ["<列名1>", "<列名2>", ...], "data": [["<值1>", <数值2>, ...], [["<值3>", <数值4>, ...]]]}}
   示例：{"table": {"columns": ["产品", "订单量"], "data": [["32085Lip", 245], ["76439Eye", 178]]}}

3. 图表输出（仅支持bar/line/scatter三种类型）：
   {"<图表类型>": {"columns": ["<x轴标签>", "<y轴标签>", ...], "data": [<数值1>, <数值2>, ...]}}
   示例：{"bar": {"columns": ["月份", "销量"], "data": [120, 150, 180]}}

注意：格式约束补充
- 所有JSON键名和字符串值必须用双引号（"）包围，不可用单引号。
- 数值类型（如数字、布尔值）无需加引号；空值用null表示。
- 禁止在Final Answer外添加任何额外内容（如解释、注释），否则视为格式错误。
注意：图表输出格式补充
"bar"/"line"/"scatter" 的 "data" 必须是 **列表的列表**，格式为：
[
  [<x轴值1>, <y轴值1>],
  [<x轴值2>, <y轴值2>],
  ...
]
例如：
{"bar": {
  "columns": ["furnishingstatus", "count"], 
  "data": [
    ["semi-furnished", 227], 
    ["unfurnished", 178], 
    ["furnished", 140]
  ]
}}
禁止使用字典格式（如 {"semi-furnished": 227}），否则无法生成图表。
你要处理的用户请求如下： 
"""
# api测试
#memory = ConversationBufferWindowMemory(k = 5,return_messages=True)
#print(get_chat_response("笛卡尔提出了哪些有名的定律？",memory,"sk-64dd601abf7a44368a48c293de9f536f"))
#print(get_chat_response("我的上一个问题是什么？要求一字不差",memory,"sk-64dd601abf7a44368a48c293de9f536f"))

