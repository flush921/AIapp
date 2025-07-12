import pandas as pd
import streamlit as st
from utils import dataframe_agent


def create_chart(input_data, chart_type):
    try:
        # 1. 校验输入数据是否包含必要字段
        if "columns" not in input_data or "data" not in input_data:
            st.error("图表数据缺少 'columns' 或 'data' 字段")
            return

        columns = input_data["columns"]
        data = input_data["data"]

        # 2. 处理 data 为字典的情况（转换为列表格式：[[key1, value1], [key2, value2], ...]）
        if isinstance(data, dict):
            # 将字典转换为列表，例如 {"a":1, "b":2} → [["a",1], ["b",2]]
            data = list(data.items())
            st.info("检测到字典格式数据，已自动转换为列表格式")

        # 3. 校验 data 是否为列表（确保能转换为 DataFrame）
        if not isinstance(data, list) or len(data) == 0:
            st.error("图表数据 'data' 必须是非空列表（或可转换为列表的字典）")
            return

        # 4. 创建 DataFrame（确保 columns 与 data 结构匹配）
        # 处理 data 中元素长度不一致的情况（例如部分元素缺少值）
        df_data = pd.DataFrame(data, columns=columns)

        # 5. 校验 DataFrame 结构（确保至少有两列：x轴和y轴）
        if len(df_data.columns) < 2:
            st.error("图表数据至少需要两列（x轴和y轴数据）")
            return

        # 6. 设置索引为 x 轴（columns[0] 通常是类别/标签列）
        df_data.set_index(columns[0], inplace=True)

        # 7. 根据图表类型渲染（使用 Streamlit 原生图表函数）
        if chart_type == "bar":
            st.bar_chart(df_data, use_container_width=True)
        elif chart_type == "line":
            st.line_chart(df_data, use_container_width=True)
        elif chart_type == "scatter":
            # 散点图需要明确 x 和 y 列（取前两列）
            x_col = df_data.columns[0]
            y_col = df_data.columns[1] if len(df_data.columns) > 1 else x_col
            st.scatter_chart(df_data, x=x_col, y=y_col, use_container_width=True)
        else:
            st.error(f"不支持的图表类型：{chart_type}，仅支持 bar/line/scatter")

    except Exception as e:
        # 捕获所有异常并显示，方便调试
        st.error(f"生成图表失败：{str(e)}")
        st.info(f"问题数据：columns={columns}, data={data}")  # 显示原始数据帮助定位问题


def csv_page():
    st.title("📚数据分析(CSV)智能工具")

    with st.sidebar:
        openai_api_key = st.text_input("输入Deepseek API Key：", type="password")
        st.markdown("[获取Deepseek API Key](https://platform.deepseek.com/api_keys)")

    data = st.file_uploader("上传你的数据文件（CSV格式）：", type="csv")
    if data:
        st.session_state["df"] = pd.read_csv(data)
        with st.expander("原始数据"):
            st.dataframe(st.session_state["df"])

    query = st.text_area("请输入你关于以上数据的问题，或数据提取请求，或生成以下可视化内容，散点图、折线图、条形图：")
    button = st.button("生成回答")

    if button and not openai_api_key:
        st.info("请先输入您的Deepseek API Key")
    if button and "df" not in st.session_state:
        st.info("请先上传数据文件")
    if button and openai_api_key and "df" in st.session_state:
        with st.spinner("Agent正在思考中，请稍等..."):
            response_dict = dataframe_agent(openai_api_key, st.session_state["df"], query)
            if "answer" in response_dict:
                st.write(response_dict["answer"])
            if "table" in response_dict:
                st.table(pd.DataFrame(response_dict["table"]["data"],
                                      columns=response_dict["table"]["columns"]))
            if "bar" in response_dict:
                create_chart(response_dict["bar"], "bar")
            if "line" in response_dict:
                create_chart(response_dict["line"], "line")
            if "scatter" in response_dict:
                create_chart(response_dict["scatter"], "scatter")
