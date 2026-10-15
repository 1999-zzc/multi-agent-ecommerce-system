from __future__ import annotations
import asyncio
from config import get_settings
from services.milvus_product_vector_store import MilvusProductVectorStore
from services.product_index_builder import ProductIndexBuilder
from services.product_repository import ProductRepository
from services.qwen_embedding import QwenEmbeddingClient


async def main() -> None:
    """离线构建商品 Milvus 索引。"""
    settings = get_settings()
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
    count = await ProductIndexBuilder(repository, embedding, vector_store).build_index()
    print(f"product index built: {count}")
if __name__ == "__main__":
    asyncio.run(main())
