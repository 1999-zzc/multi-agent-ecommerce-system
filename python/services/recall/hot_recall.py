from __future__ import annotations

# Product/UserProfile 是召回接口统一类型。
from models.schemas import Product, UserProfile

# ProductRepository 查询在线商品。
from services.product_repository import ProductRepository

# BaseRecall 定义统一召回接口。
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
        # 查询所有在线商品。真实生产可改成热度榜表或搜索服务。
        products = await self.product_repository.list_all()

        # 按 score、库存、新品标签排序,近似表示热门和可售。
        products.sort(
            key=lambda p: (
                p.score,
                p.stock > 0,
                "新品" in p.tags,
            ),
            reverse=True,
        )

        # 返回指定数量。
        return products[:limit]


"""  hot_recall
从数据库中列出所有的商品
然后按照score，stock，tag含有新品
"""