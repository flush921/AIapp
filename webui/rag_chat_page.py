import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import get_naive_rag_tool
import json

# RAG助手欢迎语
RAG_PAGE_INTRODUCTION = "你好，我是智能RAG助手，请问有什么可以帮助你的吗？"
# 获取RAG图
def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    tools = [KBS[k] for k in selected_kbs]
    tool_node = ToolNode(tools)

    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        # 将工具绑定到模型
        llm_with_tools = llm.bind_tools(tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    # 创建状态图
    workflow = StateGraph(MessagesState)

    # 添加节点到状态图
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    # 添加条件、普通边
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    # 设置入口
    workflow.set_entry_point("agent")
    # 创建技艺
    checkpointer = MemorySaver()
    # 编译
    app = workflow.compile(checkpointer=checkpointer)
    # 返回应用
    return app


# 处理RAG工作流响应
def graph_response(graph, input):
    for event in graph.invoke(
            {"messages": input},
            config={"configurable": {"thread_id": 42}},
            stream_mode="messages"
    ):
        if isinstance(event[0], AIMessageChunk):
            if event[0].tool_calls:
                st.session_state["rag_tool_calls"].append({
                    "status": "正在查询...",
                    "knowledge_base": event[0].tool_calls[0]["name"].replace("_knowledge_base_tool", ""),
                    "query": ""
                })
            yield event[0].content

        elif isinstance(event[0], ToolMessage):
            with st.status("正在查询...", expanded=True) as s:
                st.write(f"已调用 `{event[0].name.replace('_knowledge_base_tool', '')}` 知识库进行查询")
                st.code(event[0].content, wrap_lines=True)
                s.update(label="已完成知识库检索！", expanded=False)

            if st.session_state["rag_tool_calls"] and "content" not in st.session_state["rag_tool_calls"][-1]:
                st.session_state["rag_tool_calls"][-1].update({
                    "status": "已完成知识库检索！",
                    "content": json.loads(event[0].content)
                })
            else:
                st.session_state["rag_tool_calls"].append({
                    "status": "已完成知识库检索！",
                    "knowledge_base": event[0].name.replace("_knowledge_base_tool", ""),
                    "content": json.loads(event[0].content)
                })


# 获取RAG聊天响应流
def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS):
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS)
    return graph_response(graph=app, input=input)


# 显示聊天历史
def display_chat_history():
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        with st.chat_message(message["role"],
                             avatar=get_img_base64("icons8-博特-100.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call["status"], expanded=False):
                        st.write(f"已调用 `{tool_call['knowledge_base']}` 知识库进行查询")
            st.write(message["content"])


# 重置聊天会话
def clear_chat_history():
    st.session_state.update({
        "rag_chat_history": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
        "rag_chat_history_with_tool_call": [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}],
        "rag_tool_calls": []
    })


# RAG聊天页面主函数
def rag_chat_page():
    kbs = get_kb_names()
    KBS = {k: get_naive_rag_tool(k) for k in kbs}

    # 初始化会话状态
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [{"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    # 侧边栏配置
    with st.sidebar:
        selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)

    # 显示历史消息
    display_chat_history()

    # 底部输入区域
    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择模型加载方式", PLATFORMS)
            model = st.selectbox("请选择模型", get_llm_models(platform))
            temperature = st.slider("Temperature", 0.1, 1., 0.1)
            history_len = st.slider("历史消息长度", 1, 10, 5)

        user_input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)

    # 处理用户输入
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state["rag_chat_history"].append({"role": "user", "content": user_input})
        st.session_state["rag_chat_history_with_tool_call"].append({"role": "user", "content": user_input})

        # 获取并显示AI响应
        with st.chat_message("assistant", avatar=get_img_base64("icons8-博特-100.png")):
            response = st.write_stream(
                get_rag_chat_response(
                    platform, model, temperature,
                    st.session_state["rag_chat_history"][-history_len:],
                    selected_kbs, KBS
                )
            )

        # 更新会话状态
        st.session_state["rag_chat_history"].append({"role": "assistant", "content": response})
        st.session_state["rag_chat_history_with_tool_call"].append({
            "role": "assistant",
            "content": response,
            "tool_calls": st.session_state["rag_tool_calls"]
        })

        # 重置工具调用记录
        st.session_state["rag_tool_calls"] = []