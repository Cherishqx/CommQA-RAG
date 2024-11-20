import json
import os

import torch
import ujson
import numpy as np
import csv
from utils import extract_text_from_latex, split_text_by_punctuation
from embedding import ZhipuAIEmbeddingModel, BgeLargeZhv15

def append_to_json_file(data, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []
    existing_data.append(data)
    appen_to_file(existing_data,file_path)

def appen_to_file(existing_data,file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False)

def get_vector_from_json(file_path):
    with open(file_path, 'r') as f:
        data = ujson.load(f)
    return data

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def find_similar_chunks(query_text, results, embedding_model, top_n=5, print_content=True):
    query_embedding = embedding_model.get_embeddings(query_text)[0]
    similarity_results = []

    for item in results:
        similarity = cosine_similarity(query_embedding, item["embedding"][0])
        similarity_results.append((similarity, item["page_content"], item["chunk"]))

    similarity_results.sort(reverse=True, key=lambda x: x[0])
    top_results = similarity_results[:top_n]

    if print_content:
        print(top_results)
        for score, text, chunk_id in top_results:
            print(f"相似度分数: {score:.4f}, Chunk {chunk_id}: {text}")

    # if csv_file_path:
    #     with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    #         writer = csv.writer(file)
    #         for score, text, chunk_id in top_results:
    #             writer.writerow([score, chunk_id, text])

    return top_results

def load_embedding(LaTex_path, json_path):
    extracted_text = extract_text_from_latex(LaTex_path)
    chunks = split_text_by_punctuation(extracted_text)
    results = []

    vectors = get_vector_from_json(json_path)
    for i, chunk in enumerate(chunks):
        vector = vectors[i]
        print(vector)
        result = {
            'chunk': i + 1,
            'page_content': chunk,
            'embedding': vector,
        }
        results.append(result)
    print("向量已加载完毕")
    return results


def set_embedding(LaTex_path, json_path, embedding_model):
    extracted_text = extract_text_from_latex(LaTex_path)
    chunks = split_text_by_punctuation(extracted_text)

    for i, chunk in enumerate(chunks):
        embeddings = embedding_model.get_embeddings(chunk)
        print(embeddings)
        append_to_json_file(embeddings, json_path)

    print('Finish')


if __name__ == '__main__':
    # 初始化模型
    embedding_models = {
        "ZhipuAI": ZhipuAIEmbeddingModel(api_key=os.getenv("ZHIPUAI_API_KEY")),
        "bge-large-zh-v1.5": BgeLargeZhv15(),
        # "OtherModel OtherEmbeddingModel(api_key="your_api_key"),
    }

    #修改
    embedding_model = embedding_models["bge-large-zh-v1.5"]
    LaTex_path = '../main.tex'
    json_path = '../bge_vectors.json'
    #json_path = '../vectors.json'

    set_embedding(LaTex_path,json_path,embedding_model)
