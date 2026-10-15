from __future__ import annotations
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
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]
"""
支持单条和批量文本向量化，并返回可直接用于 Milvus 存储和检索的浮点向量
"""
