import os
from dotenv import load_dotenv
from flask import request, jsonify, render_template, Blueprint, redirect, url_for, Flask
from src.VectorBase import query_chroma
from src.embedding import ZhipuAIEmbeddingModel, BgeLargeZhv15
from src.llm import ZhipuAIChatModel

# 加载环境变量，只加载一次
load_dotenv()

# 初始化模型，只初始化一次
embedding_models = {
    "ZhipuAI": ZhipuAIEmbeddingModel(api_key=os.getenv("ZHIPUAI_API_KEY")),
    "bge-large-zh-v1.5": BgeLargeZhv15(),
    # "OtherModel": OtherEmbeddingModel(api_key="your_api_key"),
}

chat_model = ZhipuAIChatModel(api_key=os.getenv("ZHIPUAI_API_KEY"))

# 用于存储对话历史记录
conversation_history = []

chat1 = Blueprint('chat1', __name__)


# ------------------- 聊天相关接口 -------------------
@chat1.route('/')
def index():
    return render_template('index.html')


@chat1.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print(data)
    user_message = data.get("message", "")
    chat_history = data.get("history", [])
    # if not user_message:
    #     return jsonify({"error": "Message cannot be empty"}), 400
    reply = llm_reply(user_message, chat_history,"glm-4-flash", 0.8, 4096, "bge-large-zh-v1.5")
    return jsonify({"reply": reply})

@chat1.route('/chatt', methods=['POST'])
def chatt():
    data = request.get_json()
    print(data)
    user_message = data.get("message", "")
    chat_history = data.get("history", [])
    # if not user_message:
    #     return jsonify({"error": "Message cannot be empty"}), 400
    reply = llm_reply(user_message, chat_history,"glm-4-flash", 0.8, 4096, "bge-large-zh-v1.5")
    return jsonify({"result": 0 ,"content":reply})

def llm_reply(user_input, chat_history,model_dropdown, temperature_slider, maximum_token_slider, embedding_model_name):
    embedding_model = embedding_models[embedding_model_name]
    print("2")
    similarity_chunks = query_chroma(user_input, embedding_model, top_n=3,print_content=True)
    #similarity_chunks = find_similar_chunks(user_input, results, embedding_model, top_n=3, print_content=True)
    print("3")
    context = similarity_chunks
    # 调用 ZhipuAI 的 API 进行对话生成
    response = chat_model.generate_reply(user_input, chat_history,context, model_dropdown, temperature_slider, maximum_token_slider)

    return response