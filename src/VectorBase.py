import json
import ujson
import numpy as np
import csv
from utils import extract_text_from_latex, split_text_by_punctuation
from embedding import ZhipuAIEmbeddingModel

def append_to_json_file(data, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []

    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False)

def get_vector_from_json(file_path, index):
    with open(file_path, 'r') as f:
        data = ujson.load(f)
        vector = data[index]
    return vector

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def find_similar_chunks(query_text, results, embedding_model, top_n=5, print_content=True, csv_file_path=None):
    query_embedding = embedding_model.get_embeddings(query_text)[0]
    similarity_results = []

    for item in results:
        similarity = cosine_similarity(query_embedding, item["embedding"][0])
        similarity_results.append((similarity, item["page_content"], item["chunk"]))

    similarity_results.sort(reverse=True, key=lambda x: x[0])
    top_results = similarity_results[:top_n]

    if print_content:
        for score, text, chunk_id in top_results:
            print(f"相似度分数: {score:.4f}, Chunk {chunk_id}: {text}")

    if csv_file_path:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Similarity Score', 'Chunk ID', 'Content'])
            for score, text, chunk_id in top_results:
                writer.writerow([score, chunk_id, text])

    return top_results


def load_embedding(LaTex_path, json_path):
    # embedding_model = embedding_models[embedding_model_name]
    extracted_text = extract_text_from_latex(LaTex_path)
    chunks = split_text_by_punctuation(extracted_text)
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


    for i, chunk in enumerate(chunks):
        vector = get_vector_from_json(json_path, i)
        print(vector)
        result = {
            'chunk': i + 1,
            'page_content': chunk,
            'embedding': vector,
        }
        results.append(result)
    print("向量已加载完毕")
    return results
