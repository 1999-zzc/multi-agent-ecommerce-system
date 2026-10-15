from __future__ import annotations
from models.schemas import Product, UserProfile
from services.milvus_product_vector_store import MilvusProductVectorStore
from services.product_repository import ProductRepository
from services.qwen_embedding import QwenEmbeddingClient
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
        query_text = self._build_query_text(user_profile)
        query_vector = await self.embedding_client.embed_text(query_text)
        filter_expr = self._build_filter_expr(user_profile)
        product_ids = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
            filter_expr=filter_expr,
        )
        return await self.product_repository.get_by_ids(product_ids)

    def _build_query_text(self, user_profile: UserProfile | None) -> str:
        if not user_profile:
            return "适合电商首页推荐的热门数码商品"
        categories = ",".join(user_profile.preferred_categories) or "数码商品"
        tags = ",".join(str(v) for v in user_profile.real_time_tags.values())
        return f"用户偏好类目:{categories}; 用户实时兴趣:{tags}"

    def _build_filter_expr(self, user_profile: UserProfile | None) -> str:
        filters = ['status == "online"']
        if user_profile:
            low, high = user_profile.price_range
            filters.append(f"price >= {low}")
            filters.append(f"price <= {high}")
        return " and ".join(filters)
"""
1. 根据用户画像中的偏好类目和实时标签拼成 query，没有画像时使用通用推荐文本兜底。
2. 通过千问 Embedding 把 query 转成向量，同时根据 price_range 和 status 构造 Milvus 过滤条件。
3. 使用 query_vector 在 Milvus 中做相似度召回，并结合过滤条件过滤，得到 product_id 列表。
4. 再根据 product_id 回 MySQL 查询完整商品详情，最终返回 Product 列表。
"""
