# app_streamlit.py

import streamlit as st
from core import get_answer

# 初始化 session_state 中的聊天记录
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 页面设置
st.set_page_config(page_title="智能问答系统", page_icon="🤖")
st.title("🧠 政府采购智能问答助手")

# with st.sidebar:
#     use_search = st.checkbox("开启联网搜索", value=False)
# 显示历史聊天
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

with st.sidebar:
    st.write("## 支持作者")
    st.image("C:/Users/wtx/Desktop/mm_facetoface_collect_qrcode_1750217393422.png")
with st._bottom:
# with st.container():
    # 接收用户输入
    question = st.chat_input("请输入你的问题...",)
    # use_search = st.checkbox("联网搜索", value=False)
    use_search = st.toggle("联网搜索")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # 构造历史内容字符串，最多保留最近5轮
    history = "\n".join([
        f"用户：{q}\n助手：{a}" for q, a in st.session_state.chat_history # [-5:]
    ])

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                answer = get_answer(question=question, history=history, use_search=use_search)
            except Exception as e:
                answer = f"出错了：{str(e)}"
            st.markdown(answer)

    # 更新历史
    st.session_state.chat_history.append((question, answer))
