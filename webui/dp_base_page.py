import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from utils import get_chat_response_dp

#sk-64dd601abf7a44368a48c293de9f536f
def dp_base_page():
    st.title("😶‍🌫️Deepseek Clone")

    with st.sidebar:
        dp_api_key = st.text_input("输入Deepseek API Key：", type="password")
        st.markdown("[获取Deepseek API Key](https://platform.deepseek.com/api_keys)")


    # 初始化会话状态（确保所有需要的键都被初始化）
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferWindowMemory(k=10, return_messages=True)
        st.session_state["messages"] = [{"role": "ai", "content": "你好！请选择上方的模型开始对话"}]
        # 初始化selected_model键
        st.session_state["selected_model"] = "deepseek-V3"

    # 模型选择下拉菜单
    model_options = ["deepseek-V3", "deepseek-R1"]
    selected_model = st.selectbox(
        "选择模型",
        model_options,
        key="model_selectbox",
        format_func=lambda x: f"💭 {x}"
    )

    # 确保selected_model在会话状态中（处理可能的初始化顺序问题）
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = selected_model

    # 更新当前选择的模型
    if selected_model != st.session_state["selected_model"]:
        st.session_state["selected_model"] = selected_model
        model_tip = {
            "deepseek-V3": "已切换到Deepseek V3模型，继续提问吧！",
            "deepseek-R1": "已切换到Deepseek R1推理模型，继续提问吧~"
        }
        # 不重置消息列表，只添加模型切换提示
        st.session_state["messages"].append({"role": "ai", "content": model_tip[selected_model]})

    # 显示消息历史
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])

    # 聊天输入框
    prompt = st.chat_input(key="single_chat_input", placeholder=f"向{selected_model}提问...")

    if prompt:
        if not dp_api_key:
            st.info("请先输入您的Deepseek API Key")
            st.stop()

        st.session_state["messages"].append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)

        with st.spinner(f"{selected_model}正在思考..."):
            response = get_chat_response_dp(
                prompt,
                st.session_state["memory"],
                dp_api_key,
                selected_model
            )

        msg = {"role": "ai", "content": response}
        st.session_state["messages"].append(msg)
        st.chat_message("ai").write(response)