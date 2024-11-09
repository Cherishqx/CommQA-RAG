from abc import ABC, abstractmethod

import numpy as np
import torch
from sympy.stats.sampling.sample_numpy import numpy
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings
from zhipuai import ZhipuAI
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, text):
        pass

class ZhipuAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def get_embeddings(self, text):
        response = self.client.embeddings.create(
            input=text,
            model="embedding-3",
        )
        embeddings = [data.embedding for data in response.data]
        print(len(embeddings[0]))
        return embeddings


class BgeLargeZhv15(EmbeddingModel):
    def __init__(self):
        super().__init__()
        # 定义模型路径
        self.chat = None
        local_model_path = "../bge-large-zh-v1.5"

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.model = AutoModel.from_pretrained(local_model_path)

    def get_embeddings(self, text):
        # 使用 tokenizer 编码输入文本
        inputs = self.tokenizer(text, return_tensors="pt")

        # 将输入传递给模型并获取输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取嵌入信息（这里使用池化后的句子嵌入）
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 可以选择不同的池化方式
        embeddings = embeddings.numpy()
        embeddings = self.adjust_embedding_dimension(embeddings)
        return embeddings  # 返回嵌入向量

    def adjust_embedding_dimension(self,embedding, target_dim=2048):
        current_dim = embedding.shape[-1]
        if current_dim < target_dim:
            padding = np.zeros((embedding.shape[0], target_dim - current_dim))
            embedding = np.concatenate((embedding, padding), axis=-1)
        return embedding

# 可以在这里添加其他模型的实现，例如：
# class OtherEmbeddingModel(EmbeddingModel):
#     def __init__(self, api_key):
#         self.client = OtherAPI(api_key=api_key)
#
#     def get_embeddings(self, text):
#         response = self.client.embeddings.create(input=text)
#         embeddings = [data.embedding for data in response.data]
#         return embeddings