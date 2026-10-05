from __future__ import annotations

# OpenAI 是官方 Python SDK。千问 DashScope 提供 OpenAI 兼容接口,所以可以复用它。
from openai import AsyncOpenAI


class QwenEmbeddingClient:
    """
    千问 Embedding 客户端。

    作用:
    1. 输入商品文本或用户查询文本
    2. 调用千问 embedding 模型
    3. 返回 Milvus 可以存储/检索的 float 向量
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        # AsyncOpenAI 是异步客户端,适合 FastAPI/Agent 的 async 流程。
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 保存使用的 embedding 模型名称。
        self.model = model

    async def embed_text(self, text: str) -> list[float]:
        # 单条文本向量化,用于用户 query 或用户画像。
        vectors = await self.embed_texts([text])

        # 返回第一条向量。
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 调用千问 OpenAI 兼容 embedding 接口。
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        # 按接口返回顺序取出 embedding。
        return [item.embedding for item in response.data]

"""
支持单条和批量文本向量化，并返回可直接用于 Milvus 存储和检索的浮点向量
"""