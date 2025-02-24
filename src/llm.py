from abc import ABC, abstractmethod
from zhipuai import ZhipuAI

from src.embedding import BgeLargeZhv15


class ChatModel(ABC):
    @abstractmethod
    def generate_reply(self, user_input,history, context, model, temperature, max_tokens):
        pass

class ZhipuAIChatModel(ChatModel):
    def __init__(self, api_key):
        self.client = ZhipuAI(api_key=api_key)

    def generate_reply(self, user_input,history, context, model, temperature, max_tokens):
        messages = [
            {"role": "system",
             "content": "你是一位文档问答助手，你基于“文档内容”回答用户的问题。文档内容如下: %s" % context},
        ]
        messages.extend(history)  # 添加历史记录
        messages.append({"role": "user", "content": user_input})
        print(messages)# 添加当前用户消息
        response = self.client.chat.completions.create(
            messages=messages,
            stream=False,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        gpt_response = response.choices[0].message.content

        return gpt_response

# 可以在这里添加其他模型的实现，例如：
# class OtherChatModel(ChatModel):
#     def __init__(self, api_key):
#         self.client = OtherAPI(api_key=api_key)
#
#     def generate_reply(self, user_input, context, model, temperature, max_tokens):
#         response = self.client.chat.completions.create(
#             messages=[...],
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )
#         gpt_response = response.choices[0].message.content
#         return gpt_response