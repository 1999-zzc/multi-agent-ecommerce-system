# models 包的统一出口。
# 外部可以写 `from models import Product`,不用关心具体类定义在 schemas.py。
from .schemas import (
    UserProfile,
    Product,
    RecommendationRequest,
    RecommendationResponse,
    AgentResult,
    UserProfileResult,
    ProductRecResult,
    SubsidizedProduct,
    SubsidyDecisionResult,
    InventoryResult,
)

# 控制 `from models import *` 时导出的数据结构。
__all__ = [
    "UserProfile",
    "Product",
    "RecommendationRequest",
    "RecommendationResponse",
    "AgentResult",
    "UserProfileResult",
    "ProductRecResult",
    "SubsidizedProduct",
    "SubsidyDecisionResult",
    "InventoryResult",
]
