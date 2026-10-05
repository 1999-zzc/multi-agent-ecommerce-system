from __future__ import annotations

# Product/UserProfile 是召回接口统一类型。
from models.schemas import Product, UserProfile

# ProductRepository 查询商品。
from services.product_repository import ProductRepository

# BaseRecall 定义统一召回接口。
from .base import BaseRecall


class RuleRecall(BaseRecall):
    """基于业务规则的召回。"""

    # 简单规则: 偏好手机时,补充配件和耳机。
    RELATED_CATEGORIES = {
        "手机": {"配件", "耳机", "穿戴"},
        "平板": {"配件", "耳机"},
        "笔记本": {"配件", "显示器", "存储"},
        "游戏机": {"配件", "显示器", "耳机"},
    }

    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository

    async def recall(
        self,
        user_profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        # 没有画像时,规则召回无法判断关联类目。
        if not user_profile:
            return []

        # 根据用户偏好类目找到关联类目。
        related_categories: set[str] = set()
        for category in user_profile.preferred_categories:
            related_categories.update(self.RELATED_CATEGORIES.get(category, set()))

        # 没有命中规则时返回空。
        if not related_categories:
            return []

        # 查询在线商品,筛选关联类目商品。
        products = await self.product_repository.list_all()
        matched = [
            product
            for product in products
            if product.category in related_categories and product.stock > 0
        ]

        # 返回指定数量。
        return matched[:limit]


"""
1. 定义 RELATED_CATEGORIES 业务规则表，比如用户偏好“手机”，就扩展出“配件、耳机、穿戴”等关联类目。
2. 接收 user_profile，如果没有用户画像就直接返回空列表；如果有，就遍历 preferred_categories，根据规则表找到所有关联类目，并用 set 去重。
3. 通过 ProductRepository 中的category 作为筛选，找出关联类目且 stock > 0 的商品。
4. 最后按照 limit 截取指定数量的商品，作为规则召回结果返回。
"""