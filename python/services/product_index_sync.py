from __future__ import annotations
from services.product_index_builder import ProductIndexBuilder
from services.milvus_product_vector_store import MilvusProductVectorStore
from services.product_repository import ProductRepository
from services.qwen_embedding import QwenEmbeddingClient


class ProductIndexSync:
    """商品增量索引同步器。"""

    def __init__(
        self,
        product_repository: ProductRepository,
        embedding_client: QwenEmbeddingClient,
        vector_store: MilvusProductVectorStore,
    ):
        self.product_repository = product_repository
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    async def handle_product_event(self, event_type: str, product_id: str) -> None:
        """
        处理商品变化事件。

        product_created/product_updated:
        - 查询商品最新详情
        - 重新生成 embedding
        - upsert 到 Milvus

        product_deleted/product_offline:
        - 从 Milvus 删除向量,避免继续被召回
        """
        if event_type in {"product_deleted", "product_offline"}:
            self.vector_store.delete_product(product_id)
            return
        if event_type in {"product_created", "product_updated"}:
            product = await self.product_repository.get_by_id(product_id)
            if not product:
                return
            text = ProductIndexBuilder.product_to_embedding_text(product)
            vector = await self.embedding_client.embed_text(text)
            self.vector_store.upsert_products([product], [vector])
            return
        return
"""
如果是 product_deleted 或 product_offline
- 直接根据 product_id 从 Milvus 删除这个商品的向量
- 避免已经删除或下架的商品继续被召回

如果是 product_created 或 product_updated
- 先根据 product_id 从数据库查最新商品信息
- 把商品字段重新拼成 embedding 文本
- 调 embedding 模型生成新的向量
- 再通过 upsert_products() 写入 Milvus
- 如果原来没有这个商品，就是新增；如果已经有，就是更新
"""
