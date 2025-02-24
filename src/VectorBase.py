import os
from .utils import extract_text_from_latex, split_text_by_punctuation
from .embedding import ZhipuAIEmbeddingModel, BgeLargeZhv15
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# 获取或创建集合（collection），集合名称可自定义/记得修改
collection = client.get_or_create_collection("bge_document_chunks")#bge_document_chunks

def set_embedding(LaTex_path,embedding_model):
    extracted_text = extract_text_from_latex(LaTex_path)
    chunks = split_text_by_punctuation(extracted_text)

    # 调试：打印准备存入的文本摘要
    for i, chunk in enumerate(chunks):
        print(f"存入Chunk {i + 1} (长度 {len(chunk)}): {chunk[:100]}...")

    embeddings = [embedding_model.get_embeddings(chunk)[0] for chunk in chunks]
    add_chunks_to_chroma(chunks, embeddings)
    print('Finish')

####################

def add_chunks_to_chroma(chunks, embeddings):
    """
    将文本块和对应的嵌入添加到 Chroma 集合中。
    参数：
      chunks: 文本块列表（每个块为字符串）
      embeddings: 嵌入向量列表（每个向量为 list/np.array）
    """
    # 为每个文档分配一个唯一 id
    ids = [str(i) for i in range(len(chunks))]
    # 可选：存储一些元数据（比如分块编号）
    metadatas = [{"chunk": i + 1} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )


def query_chroma(query_text, embedding_model, top_n=5,print_content=True):
    """
    根据查询文本在 Chroma 中进行向量检索。
    返回结果中包括：文档、元数据和相似度距离（Chroma 返回的 distance 越小表示越相似）。
    """
    # 获取查询文本的嵌入（注意确保返回的类型为 list 或 np.array）
    query_embedding = embedding_model.get_embeddings(query_text)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )

    # Chroma 返回的 results 为字典格式，例如：
    # results = {
    #   "ids": [[...]],
    #   "documents": [[...]],
    #   "metadatas": [[...]],
    #   "distances": [[...]]
    # }
    if print_content==True:
        for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            print(f"Chunk {meta['chunk']} - 距离：{distance}，内容：{doc}")

    return "\n".join([f"Chunk {meta['chunk']} (距离: {distance}): {doc}" for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])])


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

    set_embedding(LaTex_path,embedding_model)
