import os
import time
import streamlit as st
from utils import PLATFORMS, get_embedding_models, get_kb_names, get_embedding_model
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import openai


# 带重试的嵌入函数，处理RateLimitError
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((openai.RateLimitError,)),
)
def retry_embed_documents(embedding_func, texts):
    return embedding_func.embed_documents(texts)


def knowledge_base_page():
    # 初始化当前选择的知识库
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = ""

    st.title("行业知识库")
    kb_names = get_kb_names()

    # 选择已有知识库或创建新库
    selected_kb = st.selectbox(
        "请选择知识库",
        ["新建知识库"] + kb_names,
        index=kb_names.index(st.session_state["selected_kb"]) + 1
        if st.session_state["selected_kb"] in kb_names else 0
    )

    # 新建知识库流程
    if selected_kb == "新建知识库":
        with st.status("知识库配置", expanded=True) as s:
            cols = st.columns(2)
            kb_name = cols[0].text_input("请输入知识库名称", placeholder="请使用英文，如：ai_information(3-18个字符)")
            vs_type = cols[1].selectbox("请选择向量库类型", ["Chroma"])
            st.text_area("请输入知识库描述", placeholder="如：介绍企业基本信息")

            cols = st.columns(2)
            platform = cols[0].selectbox("请选择Embedding模型加载方式", PLATFORMS)
            embedding_models = get_embedding_models(platform)
            embedding_model = cols[1].selectbox("请选择Embedding模型", embedding_models)

            if st.button("创建知识库"):
                if not kb_name.strip():
                    st.error("知识库名称不能为空")
                    s.update(state="error")
                    st.stop()

                # 创建知识库目录结构
                kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
                kb_path = os.path.join(kb_root, kb_name)
                file_storage_path = os.path.join(kb_path, "files")
                vs_path = os.path.join(kb_path, "vectorstore")

                if os.path.exists(kb_path):
                    st.error("知识库已存在")
                    s.update(state="error")
                    st.stop()

                os.makedirs(file_storage_path, exist_ok=True)
                os.makedirs(vs_path, exist_ok=True)

                st.success("创建知识库成功")
                s.update(label=f'已创建知识库"{kb_name}"', expanded=False)
                st.session_state["selected_kb"] = kb_name
                st.rerun()

    # 已有知识库 - 文件上传与处理
    else:
        kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
        kb_path = os.path.join(kb_root, selected_kb)
        file_storage_path = os.path.join(kb_path, "files")
        vs_path = os.path.join(kb_path, "vectorstore")

        with st.status("上传文件至知识库", expanded=True) as s:
            files = st.file_uploader("请上传文件", type=["md", "pdf"], accept_multiple_files=True)
            if st.button("上传"):
                # 保存上传的文件
                for file in files:
                    with open(os.path.join(file_storage_path, file.name), "wb") as f:
                        f.write(file.getvalue())

                # 文档加载与处理
                from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

                def load_file(file_path):
                    if file_path.endswith(".md"):
                        return TextLoader(file_path, autodetect_encoding=True)
                    elif file_path.endswith(".pdf"):
                        return PyPDFLoader(file_path)
                    raise ValueError(f"不支持的文件类型: {file_path}")
                # 仅加载本次上传的文件
                loader = DirectoryLoader(
                    file_storage_path,
                    glob=[f"**/{file.name}" for file in files],
                    show_progress=True,
                    use_multithreading=True,
                    loader_cls=load_file,
                )
                # 文本分块处理
                docs_list = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
                )
                doc_splits = text_splitter.split_documents(docs_list)
                # 添加文档源信息
                for doc in doc_splits:
                    doc.page_content = doc.metadata["source"] + "\n\n" + doc.page_content
                # 向量存储处理
                import chromadb.api
                chromadb.api.client.SharedSystemClient.clear_system_cache()
                vectorstore = Chroma(
                    collection_name=selected_kb,
                    embedding_function=get_embedding_model(platform_type="OpenAI"),
                    persist_directory=vs_path,
                )
                # 分批处理文档并嵌入向量库
                batch_size = 135
                total = len(doc_splits)
                if total == 0:
                    st.warning("未检测到有效文档内容")
                else:
                    progress_bar = st.progress(0, text="正在处理文档（0/%d）" % total)
                    for i in range(0, total, batch_size):
                        batch = doc_splits[i:i + batch_size]
                        batch_texts = [doc.page_content for doc in batch]
                        batch_metadatas = [doc.metadata for doc in batch]

                        try:
                            # 调用带重试的嵌入函数
                            embeddings = retry_embed_documents(
                                embedding_func=vectorstore._embedding_function,
                                texts=batch_texts
                            )
                            vectorstore.add_texts(
                                texts=batch_texts,
                                metadatas=batch_metadatas,
                                embeddings=embeddings
                            )
                        except openai.RateLimitError as e:
                            st.error(f"超过API速率限制，多次重试失败：{str(e)}")
                            break

                        progress = min((i + batch_size) / total, 1.0)
                        progress_bar.progress(progress,
                                              text="正在处理文档片段（%d/%d）" % (min(i + batch_size, total), total))

                    progress_bar.empty()
                    st.success("上传文件成功")