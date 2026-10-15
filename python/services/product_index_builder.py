from __future__ import annotations
from models.schemas import Product
from services.milvus_product_vector_store import MilvusProductVectorStore
from services.product_repository import ProductRepository
from services.qwen_embedding import QwenEmbeddingClient


class ProductIndexBuilder:
    """
    离线商品索引构建器。

    它应该由离线任务、定时任务或运维脚本触发,
    不应该在用户在线推荐请求中执行。
    """

    def __init__(
        self,
        product_repository: ProductRepository,
        embedding_client: QwenEmbeddingClient,
        vector_store: MilvusProductVectorStore,
    ):
        self.product_repository = product_repository
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    async def build_index(self) -> int:
        """
        全量构建商品向量索引。

        返回写入 Milvus 的商品数量。
        """
        products = await self.product_repository.list_all()
        if not products:
            return 0
        texts = [self.product_to_embedding_text(product) for product in products]
        vectors = await self.embedding_client.embed_texts(texts)
        self.vector_store.upsert_products(products, vectors)
        return len(products)

    @staticmethod
    def product_to_embedding_text(product: Product) -> str:
        """把商品结构化字段拼成 embedding 文本。"""
        return (
            f"商品名称:{product.name}; "
            f"类目:{product.category}; "
            f"品牌:{product.brand}; "
            f"价格:{product.price}; "
            f"标签:{','.join(product.tags)}"
        )
"""
1. 从数据库查询所有在线商品。
2. 把商品转换成适合 embedding 的文本。
3. 调用千问生成商品向量。
4. 写入 Milvus。
5. 返回本次写入数量。
"""
