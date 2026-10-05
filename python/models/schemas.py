from __future__ import annotations

# datetime 用于响应里的 timestamp 字段,记录推荐响应生成时间。
from datetime import datetime

# Enum 用来定义固定的用户分群枚举,避免到处使用容易写错的裸字符串。
from enum import Enum

# Any 表示任意类型,适合 data/context 这类灵活字典。
from typing import Any

# BaseModel 是 Pydantic 的数据模型基类。
# Field 用于设置默认工厂,避免 list/dict 作为可变默认值带来的问题。
from pydantic import BaseModel, Field, SerializeAsAny


class UserSegment(str, Enum):
    """用户分群枚举。继承 str 后,枚举值可以像普通字符串一样序列化到 JSON。"""

    # 新用户。
    NEW_USER = "new_user"

    # 活跃用户。
    ACTIVE = "active"

    # 高价值用户,比如消费金额高或购买频率高。
    HIGH_VALUE = "high_value"

    # 价格敏感用户,更关注折扣和性价比。
    PRICE_SENSITIVE = "price_sensitive"

    # 流失风险用户,最近活跃度下降或长时间未购买。
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    """用户画像。由 UserProfileAgent 生成,供推荐和文案 Agent 使用。"""

    # 用户唯一 ID。
    user_id: str

    # 可选基础属性。当前推荐流程不强依赖这些字段。
    age: int | None = None
    gender: str | None = None
    city: str | None = None

    # 用户分群列表。一个用户可能同时属于多个分群。
    segments: list[UserSegment] = Field(default_factory=list)

    # 偏好类目,例如 ["手机", "耳机"]。
    preferred_categories: list[str] = Field(default_factory=list)

    # 可接受价格区间,用于推荐排序时判断价格匹配度。
    price_range: tuple[float, float] = (0.0, 10000.0)

    # 最近浏览和购买记录,可作为画像或推荐上下文。
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)

    # RFM 得分: Recency/Frequency/Monetary,用来描述用户价值。
    rfm_score: dict[str, float] = Field(default_factory=dict)

    # 实时标签,例如活跃时段、偏好风格等非固定字段。
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    """商品对象。贯穿召回、重排、库存过滤和最终响应。"""

    # 商品唯一 ID。
    product_id: str

    # 商品展示名称。
    name: str

    # 商品类目,用于偏好匹配和多样性控制。
    category: str

    # 商品价格。
    price: float

    # 下面这些字段有默认值,方便示例数据和部分接口只传核心字段。
    description: str = ""
    brand: str = ""
    seller_id: str = ""

    # 商品库存。库存 Agent 会用它判断商品是否可售、是否低库存。
    stock: int = 0

    # 商品标签,例如 ["新品", "旗舰"]。
    tags: list[str] = Field(default_factory=list)

    # 推荐分数预留字段。
    score: float = 0.0

    # 商品图片 URL 预留字段。
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

    # 发生行为的用户。
    user_id: str

    # 行为类型由枚举约束，避免把 click 写成 clcik 这类拼写错误。
    behavior_type: BehaviorType

    # 本次行为关联的商品。
    item_id: str

    # 不同行为携带的额外数据，例如推荐位、来源、价格、购买数量和金额。
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationEventType(str, Enum):
    """用于衡量推荐效果的后续转化行为。"""

    CLICK = "click"
    CART = "cart"
    PURCHASE = "purchase"


class RecommendationFeedbackRequest(BaseModel):
    """用户对某次推荐结果产生后续行为时上报的请求。"""

    # 推荐接口返回的 request_id，用它关联曝光与后续转化。
    request_id: str

    # 用户 ID 用于校验反馈是否属于原推荐请求。
    user_id: str

    # 点击、加购或购买。
    event_type: RecommendationEventType

    # 被操作的推荐商品。
    product_id: str

    # 购买事件可带实际成交金额；其他事件可不传。
    amount: float | None = None


class RecommendationRequest(BaseModel):
    """推荐接口请求体。FastAPI 会用它自动校验入参。"""

    # 请求推荐的用户 ID。
    user_id: str

    # 推荐场景,例如首页、详情页、购物车页。当前默认 homepage。
    scene: str = "homepage"

    # 希望返回的商品数量。
    num_items: int = 10

    # 灵活上下文字段,可以传最近浏览、购买次数等额外信息。
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """所有 Agent 返回结果的公共父模型。"""

    # Agent 名称,例如 user_profile/product_rec/inventory。
    agent_name: str

    # 是否执行成功。
    success: bool = True

    # 本次执行耗时,单位毫秒,由 BaseAgent.run() 写入。
    latency_ms: float = 0.0

    # 错误信息。成功时通常为 None。
    error: str | None = None

    # 调试/扩展数据。不同 Agent 可以放自己的额外信息。
    data: dict[str, Any] = Field(default_factory=dict)

    # 结果置信度,0 到 1 之间。失败 fallback 通常为 0。
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

    # 商品基础信息。
    product_id: str
    name: str
    category: str
    brand: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    image_url: str = ""

    # 补贴前价格、补贴金额和补贴后价格。
    original_price: float
    subsidy_amount: float
    subsidized_price: float
    discount_rate: float

    # 两个归一化子分和最终综合分，权重各占 0.5。
    price_score: float
    discount_score: float
    composite_score: float

    # 命中的政策信息，便于前端展示和问题追踪。
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

    # 本次请求唯一 ID,方便日志追踪。
    request_id: str

    # 请求用户 ID。
    user_id: str

    # 补贴商品单独返回并放在普通推荐商品之前。
    subsidy_products: list[SubsidizedProduct] = Field(default_factory=list)

    # 普通个性化推荐商品。
    products: list[Product] = Field(default_factory=list)

    # A/B 实验分组。
    experiment_group: str = "control"

    # 各 Agent 的原始结果,便于调试和监控。
    # SerializeAsAny 保留各子类自己的 profile/products 等字段。
    agent_results: dict[str, SerializeAsAny[AgentResult]] = Field(default_factory=dict)

    # 整条推荐链路总耗时。
    total_latency_ms: float = 0.0

    # 响应生成时间。
    timestamp: datetime = Field(default_factory=datetime.now)
