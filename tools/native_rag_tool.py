from langchain.tools.retriever import create_retriever_tool
import os
from langchain_chroma import Chroma
from utils import get_embedding_model


# 创建知识库检索工具
def get_naive_rag_tool(vectorstore_name):
    # 初始化向量数据库chroma的
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=get_embedding_model(),
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name,
                                       "vectorstore"),
    )

    # 配置相似度阈值检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.15}
    )

    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        f"{vectorstore_name}_knowledge_base_tool",
        f"search and return information about {vectorstore_name}",
    )

    # 设置响应格式并自定义检索函数
    retriever_tool.response_format = "content"
    retriever_tool.func = lambda query: {
        f"已知内容 {inum + 1}": doc.page_content.replace(doc.metadata["source"] + "\n\n", "")
        for inum, doc in enumerate(retriever.invoke(query))
    }

    return retriever_tool