from abc import ABC, abstractmethod
from zhipuai import ZhipuAI

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
        return embeddings

# 可以在这里添加其他模型的实现，例如：
# class OtherEmbeddingModel(EmbeddingModel):
#     def __init__(self, api_key):
#         self.client = OtherAPI(api_key=api_key)
#
#     def get_embeddings(self, text):
#         response = self.client.embeddings.create(input=text)
#         embeddings = [data.embedding for data in response.data]
#         return embeddings