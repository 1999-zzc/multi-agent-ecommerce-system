from __future__ import annotations

# asyncio 用来运行异步索引构建任务。
import asyncio

# get_settings 读取数据库、千问、Milvus 配置。
from config import get_settings

# MilvusProductVectorStore 负责写入商品向量。
from services.milvus_product_vector_store import MilvusProductVectorStore

# ProductIndexBuilder 负责离线全量构建商品索引。
from services.product_index_builder import ProductIndexBuilder

# ProductRepository 负责查询真实商品数据库。
from services.product_repository import ProductRepository

# QwenEmbeddingClient 负责调用千问生成商品 embedding。
from services.qwen_embedding import QwenEmbeddingClient


async def main() -> None:
    """离线构建商品 Milvus 索引。"""
    # 读取项目配置。
    settings = get_settings()

    # 组装依赖。
    repository = ProductRepository()
    embedding = QwenEmbeddingClient(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.qwen_embedding_model,
    )
    vector_store = MilvusProductVectorStore(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dim,
    )

    # 执行离线索引构建。
    count = await ProductIndexBuilder(repository, embedding, vector_store).build_index()
    print(f"product index built: {count}")


if __name__ == "__main__":
    # 命令行运行: python scripts/build_product_index.py
    asyncio.run(main())
