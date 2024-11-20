import gradio as gr
import os
from dotenv import find_dotenv, load_dotenv
from embedding import ZhipuAIEmbeddingModel
from VectorBase import find_similar_chunks, load_embedding
from llm import ZhipuAIChatModel, ChatModel
from src.embedding import BgeLargeZhv15

# 加载环境变量
load_dotenv()
# 加载 .env 文件
#_ = load_dotenv(find_dotenv())

# 初始化模型
embedding_models = {
    "ZhipuAI": ZhipuAIEmbeddingModel(api_key=os.getenv("ZHIPUAI_API_KEY")),
    "bge-large-zh-v1.5": BgeLargeZhv15(),
    # "OtherModel OtherEmbeddingModel(api_key="your_api_key"),
}

chat_model = ZhipuAIChatModel(api_key=os.getenv("ZHIPUAI_API_KEY"))

# 用于存储对话历史记录
conversation_history = []

def llm_reply(user_input, model_dropdown, temperature_slider, maximum_token_slider, embedding_model_name):
    embedding_model = embedding_models[embedding_model_name]
    similarity_chunks = find_similar_chunks(user_input, results, embedding_model, top_n=3, print_content=True)
    context = similarity_chunks

    # 构建对话上下文
    messages = [
        {"role": "system", "content": "你是一位文档问答助手，你基于“文档内容”回答用户的问题。文档内容如下: %s" % context}
    ]
    # 添加之前的对话历史
    messages.extend(conversation_history)
    # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})

    # 调用 ZhipuAI 的 API 进行对话生成
    response = chat_model.generate_reply(user_input, context, model_dropdown, temperature_slider, maximum_token_slider)
    gpt_response = response

    # 更新对话历史
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": gpt_response})

    return [[user_input, gpt_response]]

if True:
    results = load_embedding('../main.tex',json_path='../bge_vectors.json')
else:
    results = load_embedding('../main.tex','../vectors.json')

with gr.Blocks() as demo:
    with gr.Row():
        # 左边对话栏
        with gr.Column():
            chatbot = gr.Chatbot(label="智能聊天机器人")
            user_input = gr.Textbox(label="输入框", placeholder="您好，请在这里输入你的问题")
            with gr.Row():
                user_submit = gr.Button("提交")
                clear_button = gr.Button("清除")

        # 右边参数栏
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["glm-4-0520", "glm-4v-plus"],
                value="glm-4-0520",
                label="LLM Model",
                interactive=True
            )
            temperature_slider = gr.Slider(label="Temperature",
                                           minimum=0,
                                           maximum=2,
                                           value=0.8,
                                           )
            maximum_token_slider = gr.Slider(label="Maximum Tokens",
                                             minimum=0,
                                             maximum=8192,
                                             value=4096,
                                             )
            embedding_model_dropdown = gr.Dropdown(
                choices=list(embedding_models.keys()),
                value="bge-large-zh-v1.5",
                label="Embedding Model",
                interactive=True
            )


        # 用户点击事件
        def handle_submit(user_input, model_dropdown, temperature_slider, maximum_token_slider, embedding_model_name):
            response = llm_reply(user_input, model_dropdown, temperature_slider, maximum_token_slider, embedding_model_name)
            return response

        user_submit.click(
            fn=handle_submit,
            inputs=[
                user_input,
                model_dropdown,
                temperature_slider,
                maximum_token_slider,
                embedding_model_dropdown,
            ],
            outputs=[chatbot]
        )

        # 实现记忆清除功能
        def clear_conversation():
            global conversation_history
            conversation_history = []
            return gr.update(value=[])

        clear_button.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chatbot]
        )

demo.launch()