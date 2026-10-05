from __future__ import annotations

# Product 是商品业务模型。
from models.schemas import Product

# MilvusProductVectorStore 负责写入商品向量。
from services.milvus_product_vector_store import MilvusProductVectorStore

# ProductRepository 负责从真实数据库查询商品。
from services.product_repository import ProductRepository

# QwenEmbeddingClient 负责调用千问生成 embedding。
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
        # 商品数据库访问层。
        self.product_repository = product_repository

        # 千问 embedding 客户端。
        self.embedding_client = embedding_client

        # Milvus 向量库。
        self.vector_store = vector_store

    async def build_index(self) -> int:
        """
        全量构建商品向量索引。

        返回写入 Milvus 的商品数量。
        """
        # 1. 从真实数据库查询所有在线商品。
        products = await self.product_repository.list_all()

        # 没有商品时直接返回 0。
        if not products:
            return 0

        # 2. 把商品转换成适合 embedding 的文本。
        texts = [self.product_to_embedding_text(product) for product in products]

        # 3. 调用千问生成商品向量。
        vectors = await self.embedding_client.embed_texts(texts)

        # 4. 写入 Milvus。
        self.vector_store.upsert_products(products, vectors)

        # 返回本次写入数量。
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