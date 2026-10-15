from __future__ import annotations
from models.schemas import Product, UserProfile
from services.product_repository import ProductRepository
from .base import BaseRecall


class HotRecall(BaseRecall):
    """热门商品召回。"""

    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository

    async def recall(
        self,
        user_profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        products = await self.product_repository.list_all()
        products.sort(
            key=lambda p: (
                p.score,
                p.stock > 0,
                "新品" in p.tags,
            ),
            reverse=True,
        )
        return products[:limit]
"""  hot_recall
从数据库中列出所有的商品
然后按照score，stock，tag含有新品
"""
