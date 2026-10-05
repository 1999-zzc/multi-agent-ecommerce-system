"""
用户画像Agent
- 实时特征提取：浏览/点击/购买/收藏行为 -> Redis Feature Store
- 用户分群：RFM模型 + 实时标签
- 画像合并：离线标签(T+1) + 在线标签(实时)
"""

from __future__ import annotations

# json 用来把行为数据转成字符串发给 LLM。
import json

# Any 用于表示不固定类型的参数和 feature_store。
from typing import Any

# LangChain 的消息类型。SystemMessage 放角色和规则,HumanMessage 放本次用户数据。
from langchain_core.messages import HumanMessage, SystemMessage

# ChatOpenAI 是 OpenAI 兼容接口客户端,这里用它调用 MiniMax 等兼容 OpenAI 协议的模型。
from langchain_openai import ChatOpenAI

# BaseModel/Field/field_validator 用来定义和校验 LLM 的结构化输出。
from pydantic import BaseModel, Field, ValidationError, field_validator

# 读取环境配置,比如模型名、API key、超时时间。
from config import get_settings

# 导入用户画像相关的数据结构。
from models.schemas import (
    UserProfile,
    UserProfileResult,
    UserSegment,
)

# 继承 BaseAgent,复用统一的运行入口、重试、日志和 fallback。
from .base_agent import BaseAgent

# 给 LLM 的系统提示词。
# 结构化输出由 Pydantic schema 约束,所以 prompt 只保留业务任务描述。
SYSTEM_PROMPT = """你是一个电商用户画像分析专家。

请根据用户行为数据,分析用户分群、偏好类目、价格区间、RFM分数和实时标签。
你必须返回 JSON 结构化结果,不要输出解释性文字。"""


class RFMScoreOutput(BaseModel):
    """LLM 输出的 RFM 分数结构。"""

    recency: float = Field(ge=0, le=1, description="最近购买得分,范围 0 到 1")
    frequency: float = Field(ge=0, le=1, description="购买频率得分,范围 0 到 1")
    monetary: float = Field(ge=0, le=1, description="消费金额得分,范围 0 到 1")


class UserProfileLLMOutput(BaseModel):
    """LLM 必须返回的用户画像结构。"""

    segments: list[UserSegment] = Field(
        default_factory=list,
        description="用户分群,可选值:new_user、active、high_value、price_sensitive、churn_risk",
    )
    preferred_categories: list[str] = Field(
        default_factory=list,
        description="用户偏好的商品类目,例如手机、耳机、平板",
    )
    price_range: list[float] = Field(
        default_factory=lambda: [0.0, 10000.0],
        description="用户可接受价格区间,格式为[最低价,最高价]",
    )
    rfm_score: RFMScoreOutput = Field(
        default_factory=lambda: RFMScoreOutput(
            recency=0.0,
            frequency=0.0,
            monetary=0.0,
        ),
        description="RFM分数",
    )
    real_time_tags: dict[str, str] = Field(
        default_factory=dict,
        description="实时标签,例如活跃时段、偏好风格",
    )

    @field_validator("price_range")
    @classmethod
    def validate_price_range(cls, value: list[float]) -> list[float]:
        """校验价格区间,避免模型返回长度不对或上下界异常。"""
        if len(value) != 2:
            return [0.0, 10000.0]

        low = float(value[0])
        high = float(value[1])

        if low < 0:
            low = 0.0
        if high <= low:
            high = low + 10000.0

        return [low, high]


class UserProfileAgent(BaseAgent):
    """根据用户行为数据生成结构化用户画像。"""

    def __init__(self, feature_store: Any = None):
        # 读取配置。这里包括 LLM API key、base_url、model 和 Agent 超时时间。
        settings = get_settings()

        # 初始化父类,告诉 BaseAgent 这个 Agent 的名字和超时配置。
        super().__init__(
            name="user_profile",
            timeout=settings.agent_timeout_user_profile,
        )

        # 初始化大模型客户端。
        # temperature=0.3 表示输出相对稳定,适合结构化分析任务。
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=1024,
        )

        # LangChain 结构化输出:
        # 模型输出会先映射到 UserProfileLLMOutput,再由 Pydantic 校验字段类型和范围。
        self.structured_llm = self.llm.with_structured_output(UserProfileLLMOutput)

        # feature_store 是可选的实时特征服务。
        # FastAPI 启动时会注入 Redis FeatureStore；未注入时才使用请求 context 兜底。
        self.feature_store = feature_store

    async def _execute(self, **kwargs: Any) -> UserProfileResult:
        # user_id 是必需参数,所以这里用 kwargs["user_id"]。
        # 如果没有传,会抛 KeyError,再由 BaseAgent 统一重试/兜底。
        user_id: str = kwargs["user_id"]

        # context 是额外上下文,比如最近浏览、购买次数等;不传时用空字典。
        context: dict = kwargs.get("context", {})

        # 收集用户行为。优先查 feature_store,没有就用 context 兜底。
        behavior_data = await self._collect_behavior(user_id, context)

        # 组装发给 LLM 的消息:
        # - SystemMessage: 告诉模型业务角色和任务
        # - HumanMessage: 传入当前用户的具体行为数据
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"用户ID: {user_id}\n行为数据: {json.dumps(behavior_data, ensure_ascii=False)}"),
        ]

        try:
            # 通过 LangChain structured output 调用模型。
            # 成功时 profile_output 已经是 Pydantic 校验后的对象,不再需要手写 json.loads。
            profile_output = await self.structured_llm.ainvoke(messages)
            profile_data = self._build_profile(user_id, profile_output)

            return UserProfileResult(
                success=True,
                profile=profile_data,
                data={"structured_output": profile_output.model_dump()},
                confidence=self._calc_confidence(profile_output),
            )
        except (ValidationError, ValueError, TypeError):
            # 结构化输出失败时返回默认画像,保证推荐链路不中断。
            profile_data = self._default_profile(user_id)
            return UserProfileResult(
                success=True,
                profile=profile_data,
                data={"fallback_reason": "structured_output_validation_failed"},
                confidence=0.35,
            )

    async def _collect_behavior(self, user_id: str, context: dict) -> dict:
        """
        收集用户行为数据。

        真实生产环境中通常从 Redis Feature Store、数据库或行为日志服务读取。
        这个项目里如果没有接入外部特征服务,就从 context 中拿模拟数据。
        """
        # 如果注入了 feature_store,优先读取 Redis 中已聚合的实时特征。
        if self.feature_store:
            real_time_features = await self.feature_store.get_user_features(user_id)

            # Redis 里有实时行为或离线标签时，直接交给 LLM 分析。
            if (
                real_time_features.get("has_realtime_data")
                or real_time_features.get("offline_tags")
            ):
                return real_time_features

        # Redis 尚无该用户行为或没有 feature_store 时，使用本次请求 context 构造行为数据。
        # 这样项目即使不连 Redis,也能跑通完整链路。
        return {
            "user_id": user_id,
            "recent_views": context.get("recent_views", ["手机", "耳机", "平板"]),
            "recent_purchases": context.get("recent_purchases", ["充电器"]),
            "view_count_7d": context.get("view_count_7d", 25),
            "purchase_count_30d": context.get("purchase_count_30d", 3),
            "avg_order_amount": context.get("avg_order_amount", 299.0),
            "active_hours": context.get("active_hours", [20, 21, 22]),
        }

    def _build_profile(
        self,
        user_id: str,
        output: UserProfileLLMOutput,
    ) -> UserProfile:
        """把 LLM 的结构化输出转换成项目内部 UserProfile。"""
        price_range = output.price_range
        return UserProfile(
            user_id=user_id,
            segments=output.segments or [UserSegment.ACTIVE],
            preferred_categories=output.preferred_categories,
            price_range=(
                float(price_range[0]),
                float(price_range[1]),
            ),
            rfm_score=output.rfm_score.model_dump(),
            real_time_tags=output.real_time_tags,
        )

    def _default_profile(self, user_id: str) -> UserProfile:
        """结构化输出失败时的默认画像。"""
        return UserProfile(
            user_id=user_id,
            segments=[UserSegment.ACTIVE],
            preferred_categories=[],
            price_range=(0.0, 10000.0),
            rfm_score={
                "recency": 0.0,
                "frequency": 0.0,
                "monetary": 0.0,
            },
            real_time_tags={},
        )

    def _fallback(
        self,
        latency_ms: float,
        exc: Exception,
        user_id: str = "unknown",
        **kwargs: Any,
    ) -> UserProfileResult:
        """画像 Agent 超时或外部依赖异常时，仍返回低置信度默认画像。"""
        # 与 BaseAgent 通用 fallback 相比，这里保留 profile 字段，方便下游安全读取。
        return UserProfileResult(
            success=False,
            profile=self._default_profile(user_id),
            latency_ms=latency_ms,
            error=str(exc),
            data={"fallback_reason": "agent_execution_failed"},
            confidence=0.2,
        )

    def _calc_confidence(self, output: UserProfileLLMOutput) -> float:
        """根据结构化输出完整度动态计算画像置信度。"""
        confidence = 0.4

        if output.segments:
            confidence += 0.15
        if output.preferred_categories:
            confidence += 0.15
        if output.price_range and len(output.price_range) == 2:
            confidence += 0.1
        if output.rfm_score:
            confidence += 0.1
        if output.real_time_tags:
            confidence += 0.1

        return min(confidence, 1.0)


"""
1.system_prompt
2.class rfm
3.class llm_output
4.structured_output+redis
5.collect_behavior
6.
"""
