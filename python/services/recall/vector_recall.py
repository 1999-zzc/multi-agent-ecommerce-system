from __future__ import annotations

# Product/UserProfile 是召回所需的业务对象。
from models.schemas import Product, UserProfile

# MilvusProductVectorStore 负责向量检索。
from services.milvus_product_vector_store import MilvusProductVectorStore

# ProductRepository 根据 Milvus 返回的 product_id 查询商品详情。
from services.product_repository import ProductRepository

# QwenEmbeddingClient 负责把 query 文本转成向量。
from services.qwen_embedding import QwenEmbeddingClient

# BaseRecall 定义统一召回接口。
from .base import BaseRecall


class VectorRecall(BaseRecall):
    """基于千问 Embedding + Milvus 的向量召回。"""

    def __init__(
        self,
        product_repository: ProductRepository,
        embedding_client: QwenEmbeddingClient,
        vector_store: MilvusProductVectorStore,
    ):
        self.product_repository = product_repository
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    async def recall(
        self,
        user_profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        # 根据用户画像生成召回 query。
        query_text = self._build_query_text(user_profile)

        # query 文本转 embedding。
        query_vector = await self.embedding_client.embed_text(query_text)

        # 根据价格区间构造 Milvus 过滤表达式。
        filter_expr = self._build_filter_expr(user_profile)

        # Milvus 返回相似商品 ID。
        product_ids = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            filter_expr=filter_expr,
        )

        # 回数据库查询完整商品详情。
        return await self.product_repository.get_by_ids(product_ids)

    def _build_query_text(self, user_profile: UserProfile | None) -> str:
        # 没有画像时使用通用首页推荐 query。
        if not user_profile:
            return "适合电商首页推荐的热门数码商品"

        # 有画像时,把偏好类目和实时标签拼进 query。
        categories = ",".join(user_profile.preferred_categories) or "数码商品"
        tags = ",".join(str(v) for v in user_profile.real_time_tags.values())
        return f"用户偏好类目:{categories}; 用户实时兴趣:{tags}"

    def _build_filter_expr(self, user_profile: UserProfile | None) -> str:
        # 默认只召回在线商品。
        filters = ['status == "online"']

        # 有价格区间时,把预算转成 Milvus 过滤条件。
        if user_profile:
            low, high = user_profile.price_range
            filters.append(f"price >= {low}")
            filters.append(f"price <= {high}")

        # 用 and 拼接成 Milvus 表达式。如:status == "online" and price >= 3000 and price <= 8000
        return " and ".join(filters)

"""
1. 根据用户画像中的偏好类目和实时标签拼成 query，没有画像时使用通用推荐文本兜底。
2. 通过千问 Embedding 把 query 转成向量，同时根据 price_range 和 status 构造 Milvus 过滤条件。
3. 使用 query_vector 在 Milvus 中做相似度召回，并结合过滤条件过滤，得到 product_id 列表。
4. 再根据 product_id 回 MySQL 查询完整商品详情，最终返回 Product 列表。
"""