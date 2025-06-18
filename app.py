# app.py
from flask import Flask, request, jsonify, render_template
from core import get_answer

# 内存中记录多个会话历史
session_memory = {}
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"reply": "问题不能为空。"})

    # 初始化历史记录
    chat_history = session_memory.get(session_id, [])

    try:
        # 拼接上下文历史
        history = "\n".join(
            [f"用户：{q}\n助手：{a}" for q, a in chat_history[-10:]]  # 限制最多轮数
        )

        # 获取回答
        answer = get_answer(question=question, history=history)

        # 保存历史
        chat_history.append((question, answer))
        session_memory[session_id] = chat_history

        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"reply": f"出错了：{str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
