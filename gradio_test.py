import json
import ujson
import gradio as gr
import os
import re
import pdfplumber
from zhipuai import ZhipuAI
import numpy as np
import csv

client = ZhipuAI(api_key="261825f908dc17b7d40d1bfb7dd96fcf.PLmGYOANYOdCSZwM")

pdf_path = '../北邮通信原理习题集电子版（带分割线版）33-64/bupt_Principles_of_Communication_Problem_Set.pdf'
pad_name = 'bupt_Principles_of_Communication_Problem_Set.pdf'
LaTex_path = 'main.tex'

# 用于存储对话历史记录
conversation_history = []

#
# # 前期准备
# # 从PDF中提取文本
# def extract_all_text(pdf_path):
#     """从指定的PDF文件中提取所有页面的文本，并将其作为一个字符串返回"""
#     if not os.path.exists((pdf_path)):
#         return "指定的文件不存在，请检查路径"
#
#     all_text = ""  # 初始化一个空字符串，用于存储所有文本
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             # 依次遍历PDF文档中的每一页
#             for page in pdf.pages:
#                 # 提取每一页的文本信息
#                 page_text = page.extract_text()
#                 if page_text is not None:
#                     all_text += page_text + "\n"
#                 else:
#                     all_text += "此页面没有可提取的文本" + "\n"
#     except Exception as e:
#         return f"处理 pdf 时出现错误"
#
#     return all_text

# 从LaTex格式文件中提取文本
def extract_text_from_latex(latex_path):
    """从指定的 LaTeX 文件中提取纯文本"""
    if not os.path.exists(latex_path):
        return "指定的文件不存在，请检查路径"

    all_text = ""  # 初始化一个空字符串，用于存储所有文本
    try:
        with open(latex_path, 'r', encoding='utf-8') as file:
            latex_content = file.read()

            # 移除 LaTeX 命令
            # 这里使用正则表达式移除所有 LaTeX 命令，如 \section{}, \textbf{} 等
            clean_text = re.sub(r'\$$a-zA-Z]+\{[^}]*\}', '', latex_content)  # 移除带大括号的命令
            clean_text = re.sub(r'\$$a-zA-Z]+', '', clean_text)  # 移除不带大括号的命令

            # 去除多余的空白符
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            all_text = clean_text

    except Exception as e:
        return f"处理 LaTeX 文件时出现错误: {str(e)}"

    return all_text


# 对文本进行分割
def split_text_by_punctuation(text):
    # 定义一个正则表达式，包括常见的中英文标点
    pattern = r"[---------------------------------------]+"
    # 使用正则表达式进行分割
    segments = re.split(pattern, text)
    # 过滤掉空字符串
    return [segment for segment in segments if segments]


# 将分割得到的文本embedding成向量（计算机能读懂的模式）
def get_embeddings(text):
    # 请求生成嵌入向量
    response = client.embeddings.create(
        input=text,
        model="embedding-3",  # 指定使用的模型
    )

    # 从响应中提取嵌入向量并返回
    embeddings = [data.embedding for data in response.data]
    return embeddings


# 将向量存到本地storage文件夹中的vectors.json文件中
def append_to_json_file(data, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # 打开文件以读取已有数据
            existing_data = json.load(file)  # 加载已有数据
    except FileNotFoundError:  # 如果文件不存在，创建一个空的数据列表
        existing_data = []

    existing_data.append(data)  # 将新数据添加到列表中

    with open(file_path, 'w', encoding='utf-8') as file:  # 以写入模式打开文件
        json.dump(existing_data, file, ensure_ascii=False)  # 将更新后的数据写入文件


# 从本地加载向量数据
def get_vector_from_json(file_path, index):
    with open(file_path, 'r') as f:
        data = ujson.load(f)
        vector = data[index]

    return vector


# 计算用户输入的问题于chunk列表中每一个元素的相似度，以得到最相似的chunk
def cosine_similarity(A, B):
    # 使用np.dot函数计算向量A和B的点积
    dot_product = np.dot(A, B)

    # 使用np.linalg.norm函数计算向量A的欧几里得范数
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    # 计算余弦相似度
    return dot_product / (norm_A * norm_B)


# 将数据查询定义为一个函数
def find_similar_chunks(query_text, results, top_n=5, print_content=True,
                        csv_file_path='../每个问题相似度排名前三的输出结果.csv'):
    """
    计算查询文本与文档片段的相似度，并返回/打印最相似的几个片段。

    参数：
    query_text (str): 查询文本
    results (list of dict): 包含文档片段和相应嵌入向量的列表
    top_n (int): 要返回的最相似片段的数量
    print_content (bool): 是否打印每个片段的文本内容
    csv_file_path (str): 要保存结果的CSV文件路径（如果为None，则不保存CSV）

    返回：
    list: 包含最相似片段的相似度和文本内容的列表
    """
    query_embedding = get_embeddings(query_text)[0]
    similarity_results = []

    # 计算相似度
    for item in results:
        similarity = cosine_similarity(query_embedding, item["embedding"][0])
        similarity_results.append((similarity, item["page_content"], item["chunk"]))

    # 排序并选取最相似的top_n个结果
    similarity_results.sort(reverse=True, key=lambda x: x[0])
    top_results = similarity_results[:top_n]

    # 根据选择打印结果
    if print_content:
        for score, text, chunk_id in top_results:
            print(f"相似度分数: {score:.4f}, Chunk {chunk_id}: {text}")

    # 如果提供了CSV文件路径，则将结果保存到CSV文件中
    if csv_file_path:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入标题行
            writer.writerow(['Similarity Score', 'Chunk ID', 'Content'])
            # 写入每一行数据
            for score, text, chunk_id in top_results:
                writer.writerow([score, chunk_id, text])

    return top_results


# # 提取文本
# extracted_text = extract_all_text(pdf_path)
#
# # 将文本分割
# chunks = split_text_by_punctuation(extracted_text)
# for i, segment in enumerate(chunks):
#     print("Chunk {}: {}".format(i + 1, segment))
#
# # 每个chunk得到的向量
# results = []
# for i, chunk in enumerate(chunks):
#     embedding = get_embeddings(chunk)
#     result = {
#         'chunk': i+1,
#         'page_content': chunk,
#         'embedding': embedding,
#     }
#     results.append(result)
#
# similarity_chunks = find_similar_chunks("请帮我解释一下通信原理", results, top_n=3, print_content=True)
# print(similarity_chunks)
#
# context = similarity_chunks

def load_embedding():
    # 提取文本
    extracted_text = extract_text_from_latex(LaTex_path)
    # 将文本分割
    chunks = split_text_by_punctuation(extracted_text)
    # for i, segment in enumerate(chunks):
    #     print("Chunk {}: {}".format(i + 1, segment))
    # 每个chunk得到的向量
    results = []

    # 如果有新的文件进来需要用这一段代码将向量存入本地
    # for i, chunk in enumerate(chunks):
    #     embeddings = get_embeddings(chunk)
    #     print(embeddings)
    #     append_to_json_file(embeddings, 'storage/vectors.json')
    #     result = {
    #         'chunk': i + 1,
    #         'page_content': chunk,
    #         'embedding': embeddings,
    #     }
    #     results.append(result)

    # 如果向量已经存入本地，则直接用这一段代码进行调用本地向量
    for i, chunk in enumerate(chunks):
        vector = get_vector_from_json('./vectors.json', i)
        print(vector)
        result = {
            'chunk': i + 1,
            'page_content': chunk,
            'embedding': vector,
        }
        results.append(result)
    print("向量已加载完毕")
    return results


def llm_reply(user_input, model_dropdown, temperature_slider, maximum_token_slider):

    global conversation_history
    similarity_chunks = find_similar_chunks(user_input, results, top_n=3, print_content=True)
    # print(similarity_chunks)

    context = similarity_chunks
    
    context.extend(conversation_history)
    """调用 ZhipuAI 的 API 进行对话生成"""
    response = client.chat.completions.create(
        messages=[
            {"role": "system",
             "content": "你是一位文档问答助手，你基于“文档内容”回答用户的问题。文档内容如下: %s" % context},
            {"role": "user",
             "content": user_input},
        ],
        stream=False,
        model=model_dropdown,  # 填写需要调用的模型编码
        temperature=temperature_slider,
        max_tokens=maximum_token_slider
    )
    gpt_response = response.choices[0].message.content
    
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "system", "content": gpt_response})
    
    return [[user_input, gpt_response]]

results = load_embedding()
# 进入主函数，这块是gradio网站的搭建
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

        # 用户点击事件
        user_submit.click(
            fn=llm_reply,     # 这是gradio网页的主函数（也就是进入网页运行逻辑就是llm_reply）
            inputs=[
                user_input,
                model_dropdown,
                temperature_slider,
                maximum_token_slider,
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
