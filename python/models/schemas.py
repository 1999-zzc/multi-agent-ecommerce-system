from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, SerializeAsAny


class UserSegment(str, Enum):
    """用户分群枚举。继承 str 后,枚举值可以像普通字符串一样序列化到 JSON。"""
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    """用户画像。由 UserProfileAgent 生成,供推荐和文案 Agent 使用。"""
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10000.0)
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    """商品对象。贯穿召回、重排、库存过滤和最终响应。"""
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""


class BehaviorType(str, Enum):
    """用户在电商页面中可能产生的实时行为类型。"""
    VIEW = "view"
    CLICK = "click"
    FAVORITE = "favorite"
    CART = "cart"
    PURCHASE = "purchase"


class BehaviorEventRequest(BaseModel):
    """写入 Redis 实时特征库的一条用户行为请求。"""
    user_id: str
    behavior_type: BehaviorType
    item_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationEventType(str, Enum):
    """用于衡量推荐效果的后续转化行为。"""
    CLICK = "click"
    CART = "cart"
    PURCHASE = "purchase"


class RecommendationFeedbackRequest(BaseModel):
    """用户对某次推荐结果产生后续行为时上报的请求。"""
    request_id: str
    user_id: str
    event_type: RecommendationEventType
    product_id: str
    amount: float | None = None


class RecommendationRequest(BaseModel):
    """推荐接口请求体。FastAPI 会用它自动校验入参。"""
    user_id: str
    scene: str = "homepage"
    num_items: int = 10
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """所有 Agent 返回结果的公共父模型。"""
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class UserProfileResult(AgentResult):
    """用户画像 Agent 的返回结果。"""
    agent_name: str = "user_profile"
    profile: UserProfile | None = None


class ProductRecResult(AgentResult):
    """商品推荐 Agent 的返回结果。"""
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""
    stage: str = "full"


class SubsidizedProduct(BaseModel):
    """补贴决策完成后返回给接口的商品。"""
    product_id: str
    name: str
    category: str
    brand: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    image_url: str = ""
    original_price: float
    subsidy_amount: float
    subsidized_price: float
    discount_rate: float
    price_score: float
    discount_score: float
    composite_score: float
    policy_id: str
    policy_name: str
    applicable_region: str
    policy_source: str = ""


class SubsidyDecisionResult(AgentResult):
    """补贴决策 Agent 的返回结果。"""
    agent_name: str = "subsidy_decision"
    products: list[SubsidizedProduct] = Field(default_factory=list)
    retrieved_policy_ids: list[str] = Field(default_factory=list)


class InventoryResult(AgentResult):
    """库存 Agent 的返回结果。"""
    agent_name: str = "inventory"
    available_products: list[str] = Field(default_factory=list)
    low_stock_alerts: list[dict[str, Any]] = Field(default_factory=list)
    purchase_limits: dict[str, int] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """推荐接口最终响应体。"""
    request_id: str
    user_id: str
    subsidy_products: list[SubsidizedProduct] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    experiment_group: str = "control"
    agent_results: dict[str, SerializeAsAny[AgentResult]] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
