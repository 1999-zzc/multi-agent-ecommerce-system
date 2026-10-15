from __future__ import annotations
from abc import ABC, abstractmethod
from models.schemas import Product, UserProfile


class BaseRecall(ABC):
    """所有召回器的统一父类。"""

    @abstractmethod
    async def recall(
        self,
        user_profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        """返回候选商品列表。"""
