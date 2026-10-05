from __future__ import annotations

# ABC/abstractmethod 用于定义召回器统一接口。
from abc import ABC, abstractmethod

# Product/UserProfile 是召回所需的业务对象。
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

