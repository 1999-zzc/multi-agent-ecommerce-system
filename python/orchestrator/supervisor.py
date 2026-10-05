"""Supervisor：先生成画像，再并行执行普通推荐与补贴决策。"""

from __future__ import annotations

# asyncio 用于同时运行两个互不依赖的商品分支。
import asyncio

# time/uuid 分别用于统计总耗时和生成请求追踪 ID。
import time
import uuid

# Any 用于可选的 Redis FeatureStore。
from typing import Any

# structlog 记录请求级结构化日志。
import structlog

# 三个真正需要推理或决策的业务 Agent。
from agents import (
    InventoryAgent,
    ProductRecAgent,
    SubsidyDecisionAgent,
    UserProfileAgent,
)

# 请求、响应和中间业务模型。
from models.schemas import (
    Product,
    RecommendationRequest,
    RecommendationResponse,
    SubsidizedProduct,
    UserProfile,
)

# A/B 引擎控制普通推荐是否启用 LLM 精排。
from services.ab_test import ABTestEngine

# 两个商品 Agent 共用同一个 MySQL Repository/连接池。
from services.product_repository import ProductRepository

logger = structlog.get_logger()


class SupervisorOrchestrator:
    """协调画像、普通推荐和补贴决策，并聚合最终响应。"""

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        feature_store: Any = None,
    ):
        # 用户画像先执行，为普通推荐提供个性化条件。
        self.user_profile_agent = UserProfileAgent(feature_store=feature_store)

        # 共用 Repository，避免同一进程重复创建数据库引擎。
        product_repository = ProductRepository()
        self.product_rec_agent = ProductRecAgent(product_repository=product_repository)
        self.subsidy_decision_agent = SubsidyDecisionAgent(
            product_repository=product_repository
        )
        self.inventory_agent = InventoryAgent()

        self.ab_engine = ab_engine or ABTestEngine()

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """执行一次完整推荐链路。"""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        logger.info(
            "supervisor.start",
            request_id=request_id,
            user_id=request.user_id,
            scene=request.scene,
        )

        # 相同 user_id 会稳定进入同一实验组。
        experiment = self.ab_engine.assign(request.user_id)
        use_llm_rerank = experiment.get("config", {}).get("rerank", "llm") == "llm"

        # 画像必须先完成，因为普通推荐需要它做个性化召回和精排。
        profile_result = await self.user_profile_agent.run(
            user_id=request.user_id,
            context=request.context,
        )
        user_profile: UserProfile | None = getattr(profile_result, "profile", None)
        if getattr(profile_result, "confidence", 0.0) < 0.5:
            user_profile = None

        # 普通推荐和补贴决策没有相互依赖关系，因此并行启动。
        rec_result, subsidy_result = await asyncio.gather(
            self.product_rec_agent.run(
                user_profile=user_profile,
                # 多取 4 个候选，去掉与补贴区重复的商品后仍尽量补满普通推荐。
                num_items=request.num_items + 4,
                use_llm_rerank=use_llm_rerank,
            ),
            self.subsidy_decision_agent.run(
                context=request.context,
                limit=4,
            ),
        )

        subsidy_products: list[SubsidizedProduct] = getattr(
            subsidy_result,
            "products",
            [],
        )
        recommended_products: list[Product] = getattr(rec_result, "products", [])

        # 两个并行分支完成后，统一交给库存 Agent 做最终可售校验。
        inventory_candidates = list(
            {
                product.product_id: product
                for product in [*subsidy_products, *recommended_products]
            }.values()
        )
        inventory_result = await self.inventory_agent.run(
            products=inventory_candidates
        )
        if getattr(inventory_result, "success", False):
            available_ids = set(
                getattr(inventory_result, "available_products", [])
            )
            subsidy_products = [
                product
                for product in subsidy_products
                if product.product_id in available_ids
            ]
            recommended_products = [
                product
                for product in recommended_products
                if product.product_id in available_ids
            ]

        # 补贴区已经展示的商品不在普通推荐区重复出现。
        subsidy_ids = {product.product_id for product in subsidy_products}
        final_products = [
            product
            for product in recommended_products
            if product.product_id not in subsidy_ids
        ][: request.num_items]

        total_latency = (time.perf_counter() - start) * 1000
        logger.info(
            "supervisor.complete",
            request_id=request_id,
            total_latency_ms=round(total_latency, 1),
            subsidy_product_count=len(subsidy_products),
            product_count=len(final_products),
        )

        # 字段顺序就是接口 JSON 的展示顺序：补贴商品在前，普通推荐在后。
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            subsidy_products=subsidy_products,
            products=final_products,
            experiment_group=experiment.get("group", "control"),
            agent_results={
                "user_profile": profile_result,
                "product_rec": rec_result,
                "subsidy_decision": subsidy_result,
                "inventory": inventory_result,
            },
            total_latency_ms=total_latency,
        )
