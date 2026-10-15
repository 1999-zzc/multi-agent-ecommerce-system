"""
实时特征存储服务
- Redis Sorted Set 存储用户行为序列 (score=timestamp)
- 滑动窗口计算实时特征 (1h/24h/7d)
- 离线+在线特征合并
- RFM模型计算
"""

from __future__ import annotations
import json
import time
from typing import Any
import structlog

logger = structlog.get_logger()


class FeatureStore:
    """
    基于 Redis 的实时特征存储。

    核心思想:
    - 用户行为用 Redis Sorted Set 存储
    - score 使用时间戳
    - 查询时用 zrangebyscore 做滑动窗口统计
    """

    def __init__(self, redis_client: Any = None, ttl: int = 2592000):
        self.redis = redis_client
        self.ttl = ttl

    async def record_behavior(
        self, user_id: str, behavior_type: str, item_id: str, metadata: dict | None = None
    ) -> bool:
        """记录一条用户行为事件。"""
        key = f"behavior:{user_id}:{behavior_type}"
        event_ts = time.time()
        safe_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if key not in {"item_id", "ts"}
        }
        payload = json.dumps(
            {"item_id": item_id, "ts": event_ts, **safe_metadata},
            ensure_ascii=False,
        )
        await self.redis.zadd(key, {payload: event_ts})
        await self.redis.expire(key, self.ttl)
        return True

    async def get_recent_behaviors(
        self, user_id: str, behavior_type: str, window_seconds: int = 3600
    ) -> list[dict]:
        """查询指定滑动窗口内的用户行为。"""
        if not self.redis:
            return []
        key = f"behavior:{user_id}:{behavior_type}"
        cutoff = time.time() - window_seconds
        raw_items = await self.redis.zrangebyscore(key, cutoff, "+inf")
        return [json.loads(item) for item in raw_items]

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """聚合用户实时特征,供 UserProfileAgent 使用。"""
        views_1h = await self.get_recent_behaviors(user_id, "view", 3600)
        views_24h = await self.get_recent_behaviors(user_id, "view", 86400)
        clicks_1h = await self.get_recent_behaviors(user_id, "click", 3600)
        favorites_7d = await self.get_recent_behaviors(user_id, "favorite", 604800)
        carts_7d = await self.get_recent_behaviors(user_id, "cart", 604800)
        purchases_7d = await self.get_recent_behaviors(user_id, "purchase", 604800)
        purchases_30d = await self.get_recent_behaviors(user_id, "purchase", 2592000)
        recent_view_items = [v.get("item_id", "") for v in views_24h[-20:]]
        recent_purchase_items = [p.get("item_id", "") for p in purchases_30d[-10:]]
        rfm = await self._compute_rfm(
            user_id,
            purchases_30d=purchases_30d,
            purchases_7d=purchases_7d,
        )
        profile_key = f"profile:{user_id}"
        offline_tags = {}
        if self.redis:
            raw = await self.redis.get(profile_key)
            if raw:
                offline_tags = json.loads(raw)
        return {
            "user_id": user_id,
            "view_count_1h": len(views_1h),
            "view_count_24h": len(views_24h),
            "click_count_1h": len(clicks_1h),
            "favorite_count_7d": len(favorites_7d),
            "cart_count_7d": len(carts_7d),
            "purchase_count_7d": len(purchases_7d),
            "purchase_count_30d": len(purchases_30d),
            "recent_views": recent_view_items,
            "recent_purchases": recent_purchase_items,
            "rfm": rfm,
            "offline_tags": offline_tags,
            "has_realtime_data": bool(
                views_1h
                or views_24h
                or clicks_1h
                or favorites_7d
                or carts_7d
                or purchases_7d
                or purchases_30d
            ),
        }

    async def _compute_rfm(
        self,
        user_id: str,
        purchases_30d: list[dict],
        purchases_7d: list[dict],
    ) -> dict[str, float]:
        """
        计算 RFM 得分。

        R: Recency,最近一次购买越近分越高
        F: Frequency,购买次数越多分越高
        M: Monetary,平均消费金额越高分越高

        当前数据较少,所以使用简单启发式归一化到 0-1。
        """
        if not purchases_30d:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}
        now = time.time()
        latest_ts = max(p.get("ts", 0) for p in purchases_30d)
        days_since = (now - latest_ts) / 86400
        recency = max(0.0, 1.0 - days_since / 30.0)
        frequency = min(1.0, len(purchases_7d) / 10.0)
        avg_amount = sum(p.get("amount", 100) for p in purchases_30d) / len(purchases_30d)
        monetary = min(1.0, avg_amount / 1000.0)
        return {
            "recency": round(recency, 3),
            "frequency": round(frequency, 3),
            "monetary": round(monetary, 3),
        }

    async def merge_offline_tags(self, user_id: str, tags: dict[str, Any]):
        """写入离线标签,供画像 Agent 合并使用。"""
        if not self.redis:
            return
        key = f"profile:{user_id}"
        await self.redis.set(key, json.dumps(tags), ex=self.ttl)
"""
1.record_behavior 通过用户id+type作为redis的key，把用户行为事件以 JSON 形式作为排序的member，然后以事件时间戳作为 score 存储，
从而支持后续按时间窗口查询用户的浏览、点击、购买等实时行为。
2.get_recent_behavior：通过redis返回某一时间段后用户的行为
3.get_future_behavior：返回用户的结构化特征
4.rfm r:max(0,1-/30) f:min(1,len(day7)/10),m:min(1,avg)
5.merge_offline_tags：key+tags
"""
