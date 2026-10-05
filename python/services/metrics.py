"""
监控指标收集
- Agent调用成功率 / 延迟
- 推荐CTR / CVR / GMV
- A/B测试实验指标
"""

from __future__ import annotations

# time 用于记录业务事件发生时间。
import time

# defaultdict 可以在第一次访问某个 agent_name 时自动创建 AgentMetric。
from collections import defaultdict

# dataclass 用来简洁定义指标结构;field 用于安全创建默认 list。
from dataclasses import dataclass, field

# Any 用于业务事件中的灵活字段。
from typing import Any


@dataclass
class AgentMetric:
    """单个 Agent 的调用统计。"""

    # 调用总次数。
    call_count: int = 0

    # 成功次数。
    success_count: int = 0

    # 累计耗时,用于计算平均耗时。
    total_latency_ms: float = 0.0

    # 最近错误列表。
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """成功率 = 成功次数 / 调用次数。"""
        return self.success_count / self.call_count if self.call_count else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """平均耗时 = 总耗时 / 调用次数。"""
        return self.total_latency_ms / self.call_count if self.call_count else 0.0


class MetricsCollector:
    """内存版指标收集器。生产环境可以替换成 Prometheus 或日志埋点系统。"""

    def __init__(self):
        # Agent 指标,按 agent_name 聚合。
        self._agent_metrics: dict[str, AgentMetric] = defaultdict(AgentMetric)

        # 业务事件列表,例如曝光、点击、购买、GMV 等。
        self._business_events: list[dict[str, Any]] = []

        # 保存近期推荐请求上下文，用 request_id 把曝光和后续点击/购买关联起来。
        # 当前为内存版；生产环境应写入日志系统、Redis 或数仓。
        self._recommendations: dict[str, dict[str, Any]] = {}


        """
        self._agent_metrics: dict[str, AgentMetric] = {
            "recall_agent": AgentMetric(...),
            "rerank_agent": AgentMetric(...)
        }
        self._business_events: list[dict[str, Any]] = [
            {"type": "exposure", ...},
            {"type": "click", ...},
            {"type": "purchase", ...}
        ]
        self._recommendations: dict[str, dict[str, Any]] = {
            "req_001": {
            "user_id": "u123",
            "experiment_group": "control",
            "product_ids": {"p1", "p2"}
        }
        }
        """

    def record_agent_call(self, agent_name: str, success: bool, latency_ms: float, error: str = ""):
        """记录一次 Agent 调用。"""
        # 获取该 Agent 的指标对象;不存在会自动创建。
        m = self._agent_metrics[agent_name]

        # 总调用数加 1。
        m.call_count += 1

        # 成功时成功数加 1。
        if success:
            m.success_count += 1

        # 累加耗时。
        m.total_latency_ms += latency_ms

        # 有错误信息时记录下来。
        if error:
            m.errors.append(error)

    def record_business_event(self, event_type: str, **kwargs: Any):
        """记录业务事件,例如 CTR/CVR/GMV 相关埋点。"""
        self._business_events.append({
            "type": event_type,
            "timestamp": time.time(),
            **kwargs,
        })

    def record_recommendation_exposure(
        self,
        request_id: str,
        user_id: str,
        experiment_group: str,
        product_ids: list[str],
    ) -> None:
        """记录一次推荐曝光，并保存后续转化需要的关联信息。"""
        self._recommendations[request_id] = {
            "user_id": user_id,
            "experiment_group": experiment_group,
            "product_ids": set(product_ids),
        }

        # 推荐列表里的每个商品都是一次独立曝光，用于计算按商品曝光口径的 CTR/CVR。
        for product_id in product_ids:
            self.record_business_event(
                "exposure",
                request_id=request_id,
                user_id=user_id,
                experiment_group=experiment_group,
                product_id=product_id,
            )

    def record_recommendation_event(
        self,
        request_id: str,
        user_id: str,
        event_type: str,
        product_id: str,
        amount: float | None = None,
    ) -> dict[str, Any] | None:
        """记录推荐商品的点击、加购或购买；不接受无法关联到曝光的反馈。"""
        recommendation = self._recommendations.get(request_id)

        # request_id、用户和商品都必须与原推荐曝光对应，避免把无关行为计入实验结果。
        if (
            not recommendation
            or recommendation["user_id"] != user_id
            or product_id not in recommendation["product_ids"]
        ):
            return None

        event = {
            "request_id": request_id,
            "user_id": user_id,
            "experiment_group": recommendation["experiment_group"],
            "product_id": product_id,
        }
        if amount is not None:
            event["amount"] = amount
        self.record_business_event(event_type, **event)
        return event

    def get_agent_stats(self) -> dict[str, dict[str, Any]]:
        """返回所有 Agent 的聚合统计。"""
        result = {}
        for name, m in self._agent_metrics.items():
            result[name] = {
                "call_count": m.call_count,
                "success_rate": round(m.success_rate, 4),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "recent_errors": m.errors[-5:],
            }
        return result

    def get_business_stats(self) -> dict[str, Any]:
        """按事件类型统计业务事件数量。"""
        if not self._business_events:
            return {}

        # 先按事件类型分组。
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in self._business_events:
            by_type[e["type"]].append(e)

        # 聚合曝光到购买的核心漏斗指标。
        stats = {}
        exposure_count = len(by_type.get("exposure", []))
        click_count = len(by_type.get("click", []))
        cart_count = len(by_type.get("cart", []))
        purchase_events = by_type.get("purchase", [])
        purchase_count = len(purchase_events)
        gmv = sum(float(event.get("amount", 0.0)) for event in purchase_events)

        stats["funnel"] = {
            "exposure_count": exposure_count,
            "click_count": click_count,
            "cart_count": cart_count,
            "purchase_count": purchase_count,
            "ctr": round(click_count / exposure_count, 4) if exposure_count else 0.0,
            "cvr": round(purchase_count / exposure_count, 4) if exposure_count else 0.0,
            "gmv": round(gmv, 2),
        }

        # 保留每种行为的原始次数，便于接口调试。
        for t, events in by_type.items():
            stats[t] = {"count": len(events)}
        return stats


"""
Agent 执行
   ↓
record_agent_call()
   ↓
记录：调用次数、成功次数、耗时、错误
   ↓
get_agent_stats()
   ↓
得到：成功率、平均延迟、最近错误


推荐请求产生
   ↓
record_recommendation_exposure()
   ↓
保存 request_id 对应的推荐上下文
   ↓
记录每个商品 exposure

用户后续点击 / 加购 / 购买
   ↓
record_recommendation_event()
   ↓
根据 request_id 找回原推荐
   ↓
校验用户、商品是否匹配
   ↓
record_business_event()
   ↓
保存 click / cart / purchase

最后
   ↓
get_business_stats()
   ↓
统计曝光、点击、加购、购买
   ↓
计算 CTR / CVR / GMV"""