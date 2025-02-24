import csv
import os
from dotenv import load_dotenv
from embedding import ZhipuAIEmbeddingModel
from VectorBase import find_similar_chunks, load_embedding
from llm import ZhipuAIChatModel
from src.embedding import BgeLargeZhv15

###待修改


# 加载环境变量
load_dotenv()

# 初始化模型
embedding_models = {
    "ZhipuAI": ZhipuAIEmbeddingModel(api_key=os.getenv("ZHIPUAI_API_KEY")),
    "bge-large-zh-v1.5": BgeLargeZhv15(),
    # "OtherModel OtherEmbeddingModel(api_key="your_api_key"),
}

chat_model = ZhipuAIChatModel(api_key=os.getenv("ZHIPUAI_API_KEY"))

def llm_reply(user_input, model_dropdown="glm-4-0520", temperature = 0.8, token=4096, embedding_model_name="bge-large-zh-v1.5"):
    embedding_model = embedding_models[embedding_model_name]
    similarity_chunks = find_similar_chunks(user_input, results, embedding_model, top_n=3, print_content=False)
    context = similarity_chunks

    return context

question_path = "../question.csv"

embedding_model = 'zhipu'#修改模型
match embedding_model:
    case 'zhipu':
        output_path = "../zhipu_output.csv"
        results = load_embedding('../main.tex', '../vectors.json')
        embedding_model_name = "ZhipuAI"
    case 'bge':
        output_path = "../bge_output.csv"
        results = load_embedding('../main.tex', '../bge_vectors.json')
        embedding_model_name = "bge-large-zh-v1.5"

file = csv.writer(open(output_path,'w',newline='', encoding='utf-8'))
file.writerow(['问题','回答1','回答2','回答3'])

file=csv.writer(open(output_path,'a',newline='', encoding='utf-8'))
with open(question_path,mode='r',encoding='utf-8') as f:
    reader = csv.reader(f)
    # 遍历每一行数据
    for row in reader:
        output = []
        output.append(row[0])
        response = llm_reply(row[0],embedding_model_name=embedding_model_name)
        for score, text, chunk_id in response:
            a = f"相似度分数: {score:.4f}, Chunk {chunk_id}: {text}"
            output.append(a)
        print(output)
        file.writerow(output)
