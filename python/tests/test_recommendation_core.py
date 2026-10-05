"""推荐链路核心逻辑测试，不依赖真实 Redis、数据库、Milvus 或 LLM。"""

from __future__ import annotations

import asyncio
import os
import sys

# 将 python/ 目录加入模块搜索路径，支持直接运行本文件。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from agents.inventory_agent import InventoryAgent
from agents.product_rec_agent import ProductRecAgent
from agents.user_profile_agent import UserProfileAgent
from models.schemas import AgentResult, Product
from services.feature_store import FeatureStore
from services.metrics import MetricsCollector
from services.recommendation_evaluator import evaluate_rankings, mrr, ndcg_at_k, recall_at_k


class SlowAgent(BaseAgent):
    """用于验证 BaseAgent 是否会强制取消超时协程的测试 Agent。"""

    def __init__(self):
        # 只重试一次，避免测试因指数退避等待过久。
        super().__init__(name="slow", timeout=0.01, max_retries=1)

    async def _execute(self, **kwargs) -> AgentResult:
        await asyncio.sleep(0.05)
        return AgentResult(agent_name=self.name)


class FakeRedis:
    """最小 Redis 替身，只实现 FeatureStore 本测试需要的方法。"""

    def __init__(self):
        self.sorted_sets: dict[str, list[tuple[str, float]]] = {}
        self.values: dict[str, str] = {}

    async def zadd(self, key: str, mapping: dict[str, float]):
        self.sorted_sets.setdefault(key, []).extend(mapping.items())

    async def expire(self, key: str, ttl: int):
        return True

    async def zrangebyscore(self, key: str, minimum: float, maximum: str):
        items = self.sorted_sets.get(key, [])
        return [item for item, score in sorted(items, key=lambda pair: pair[1]) if score >= minimum]

    async def get(self, key: str):
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int):
        self.values[key] = value


def test_base_agent_returns_failure_after_timeout():
    """LLM 或外部服务超时后，BaseAgent 应返回失败结果而不是无限等待。"""
    result = asyncio.run(SlowAgent().run())
    assert result.success is False
    assert "timed out" in (result.error or "")


def test_feature_store_records_behavior_and_protects_system_fields():
    """行为记录应写入 Redis，同时 metadata 不得覆盖真实商品 ID 和时间戳。"""
    async def scenario():
        store = FeatureStore(redis_client=FakeRedis())
        await store.record_behavior(
            user_id="U001",
            behavior_type="purchase",
            item_id="P001",
            metadata={"amount": 7999, "item_id": "bad-id", "ts": 0},
        )
        events = await store.get_recent_behaviors("U001", "purchase", 30)
        assert events[0]["item_id"] == "P001"
        assert events[0]["ts"] > 0
        assert events[0]["amount"] == 7999

    asyncio.run(scenario())


def test_profile_default_fallback_is_a_valid_profile():
    """结构化输出失败时，画像 Agent 的默认画像应可供下游推荐继续使用。"""
    # 不初始化 LLM 客户端，直接测试纯粹的默认画像构造逻辑。
    agent = object.__new__(UserProfileAgent)
    profile = agent._default_profile("U001")
    assert profile.user_id == "U001"
    assert profile.preferred_categories == []
    assert profile.rfm_score["recency"] == 0.0


def test_multi_recall_fusion_removes_duplicate_products():
    """同一商品被多路召回时，融合结果只能保留一份，并按融合分排序。"""
    product_a = Product(product_id="P001", name="A", category="手机", price=1000)
    product_b = Product(product_id="P002", name="B", category="耳机", price=500)
    product_c = Product(product_id="P003", name="C", category="配件", price=100)

    # _merge_candidates 不依赖实例状态，因此用未初始化对象即可测试纯融合逻辑。
    agent = object.__new__(ProductRecAgent)
    merged = agent._merge_candidates(
        ([product_a, product_b], 0.6),
        ([product_b, product_c], 0.3),
        limit=3,
    )
    ids = [product.product_id for product in merged]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"P001", "P002", "P003"}


def test_inventory_removes_out_of_stock_product():
    """库存 Agent 应过滤 stock 为 0 的商品。"""
    products = [
        Product(product_id="P001", name="有货", category="手机", price=1000, stock=10),
        Product(product_id="P002", name="无货", category="耳机", price=500, stock=0),
    ]
    result = asyncio.run(InventoryAgent().run(products=products))
    assert result.available_products == ["P001"]
    assert "P002" not in result.available_products


def test_online_metrics_link_exposure_to_conversion():
    """只有能关联到原推荐曝光的反馈才计入 CTR、CVR 和 GMV。"""
    metrics = MetricsCollector()
    metrics.record_recommendation_exposure("R001", "U001", "control", ["P001", "P002"])
    assert metrics.record_recommendation_event("R001", "U001", "click", "P001")
    assert metrics.record_recommendation_event("R001", "U001", "purchase", "P001", 7999)
    assert metrics.record_recommendation_event("R001", "U001", "click", "P999") is None

    funnel = metrics.get_business_stats()["funnel"]
    assert funnel == {
        "exposure_count": 2,
        "click_count": 1,
        "cart_count": 0,
        "purchase_count": 1,
        "ctr": 0.5,
        "cvr": 0.5,
        "gmv": 7999.0,
    }


def test_offline_ranking_metrics():
    """离线评估函数应正确计算 Recall@K、NDCG@K 和 MRR。"""
    recommended = ["P001", "P002", "P003"]
    relevant = {"P002", "P004"}
    assert recall_at_k(recommended, relevant, 2) == 0.5
    assert 0 < ndcg_at_k(recommended, relevant, 3) < 1
    assert mrr(recommended, relevant) == 0.5

    result = evaluate_rankings([(recommended, relevant)], k=2)
    assert result["recall@2"] == 0.5


if __name__ == "__main__":
    # 允许不安装 pytest，直接运行本文件也能完成基础验证。
    test_base_agent_returns_failure_after_timeout()
    test_feature_store_records_behavior_and_protects_system_fields()
    test_profile_default_fallback_is_a_valid_profile()
    test_multi_recall_fusion_removes_duplicate_products()
    test_inventory_removes_out_of_stock_product()
    test_online_metrics_link_exposure_to_conversion()
    test_offline_ranking_metrics()
    print("All recommendation core tests passed!")
