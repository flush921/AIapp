import streamlit as st
from webui import rag_chat_page, knowledge_base_page, dp_base_page, csv_page
from utils import get_img_base64


if __name__ == "__main__":
    with st.sidebar:
        st.logo(
            get_img_base64("chat_lite_icon.png"),
            size="large",
            icon_image=get_img_base64("chat_lite_icon.png"),
        )

    pg = st.navigation({
        "对话": [
            st.Page(dp_base_page, title="😶‍🌫️V3/R1DeepseekClone"),
            st.Page(csv_page, title="📚数据分析(CSV)智能工具"),
            st.Page(rag_chat_page, title="🚩知识库对话"),
        ],
        "知识库设置": [
            st.Page(knowledge_base_page, title="个人知识库", icon=":material/library_books:"),
        ]
    })
    pg.run()
