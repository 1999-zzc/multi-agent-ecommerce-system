"""LangGraph：画像 -> {普通推荐 || 补贴决策} -> 聚合。"""

from __future__ import annotations

# asyncio 用于在一个图节点内并行运行两个商品 Agent。
import asyncio

# time/uuid 用于统计图耗时和生成请求 ID。
import time
import uuid

# TypedDict 描述 LangGraph 节点共享的状态。
from typing import Any, TypedDict

# END 是结束节点，StateGraph 用来声明执行图。
from langgraph.graph import END, StateGraph

# 四个业务 Agent。
from agents import (
    InventoryAgent,
    ProductRecAgent,
    SubsidyDecisionAgent,
    UserProfileAgent,
)

# 图状态中保存的业务对象。
from models.schemas import Product, SubsidizedProduct, UserProfile

# A/B 分组和数据库仓储层。
from services.ab_test import ABTestEngine
from services.product_repository import ProductRepository


class PipelineState(TypedDict, total=False):
    """LangGraph 各节点之间传递的状态。"""

    # 请求字段。
    request_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]

    # 实验字段。
    experiment_group: str
    use_llm_rerank: bool

    # 中间与最终结果。
    user_profile: UserProfile | None
    subsidy_products: list[SubsidizedProduct]
    recommended_products: list[Product]
    final_products: list[Product]
    agent_results: dict[str, Any]

    # 耗时字段。
    total_latency_ms: float
    _start_time: float


# Agent 在 build_recommendation_graph() 中创建，避免导入模块时立即连接外部服务。
user_profile_agent: UserProfileAgent | None = None
product_rec_agent: ProductRecAgent | None = None
subsidy_decision_agent: SubsidyDecisionAgent | None = None
inventory_agent: InventoryAgent | None = None
ab_engine = ABTestEngine()


async def init_node(state: PipelineState) -> PipelineState:
    """初始化追踪信息和 A/B 实验配置。"""
    state["request_id"] = str(uuid.uuid4())
    state["_start_time"] = time.perf_counter()
    state["agent_results"] = {}

    experiment = ab_engine.assign(state["user_id"])
    state["experiment_group"] = experiment.get("group", "control")
    state["use_llm_rerank"] = (
        experiment.get("config", {}).get("rerank", "llm") == "llm"
    )
    return state


async def user_profile_node(state: PipelineState) -> PipelineState:
    """先生成画像，供普通推荐分支使用。"""
    assert user_profile_agent is not None
    result = await user_profile_agent.run(
        user_id=state["user_id"],
        context=state.get("context", {}),
    )
    profile = getattr(result, "profile", None)
    if getattr(result, "confidence", 0.0) < 0.5:
        profile = None

    state["user_profile"] = profile
    state["agent_results"]["user_profile"] = result
    return state


async def parallel_product_node(state: PipelineState) -> PipelineState:
    """并行执行 ProductRecAgent 和 SubsidyDecisionAgent。"""
    assert product_rec_agent is not None
    assert subsidy_decision_agent is not None
    num_items = state.get("num_items", 10)

    # 两个协程在同一次 gather 中启动；补贴分支不等待普通推荐分支。
    rec_result, subsidy_result = await asyncio.gather(
        product_rec_agent.run(
            user_profile=state.get("user_profile"),
            num_items=num_items + 4,
            use_llm_rerank=state.get("use_llm_rerank", True),
        ),
        subsidy_decision_agent.run(
            context=state.get("context", {}),
            limit=4,
        ),
    )

    subsidy_products = getattr(subsidy_result, "products", [])
    recommended_products = getattr(rec_result, "products", [])

    state["subsidy_products"] = subsidy_products
    state["recommended_products"] = recommended_products
    state["agent_results"]["product_rec"] = rec_result
    state["agent_results"]["subsidy_decision"] = subsidy_result
    return state


async def inventory_node(state: PipelineState) -> PipelineState:
    """对补贴商品和普通推荐商品做统一库存校验。"""
    assert inventory_agent is not None
    subsidy_products = state.get("subsidy_products", [])
    recommended_products = state.get("recommended_products", [])

    # 按商品 ID 去重后交给库存 Agent，避免同一商品重复检查。
    candidates = list(
        {
            product.product_id: product
            for product in [*subsidy_products, *recommended_products]
        }.values()
    )
    result = await inventory_agent.run(products=candidates)

    # 库存 Agent 正常返回时执行过滤；异常时保留上游结果，由其 fallback 信息告警。
    if getattr(result, "success", False):
        available_ids = set(getattr(result, "available_products", []))
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

    subsidy_ids = {product.product_id for product in subsidy_products}
    state["subsidy_products"] = subsidy_products
    state["final_products"] = [
        product
        for product in recommended_products
        if product.product_id not in subsidy_ids
    ][: state.get("num_items", 10)]
    state["agent_results"]["inventory"] = result
    return state


async def aggregate_node(state: PipelineState) -> PipelineState:
    """计算整张图的执行耗时。"""
    state["total_latency_ms"] = (
        time.perf_counter() - state.get("_start_time", time.perf_counter())
    ) * 1000
    return state


def build_recommendation_graph(
    feature_store: Any = None,
    ab_engine_instance: ABTestEngine | None = None,
) -> StateGraph:
    """构建并编译推荐状态图。"""
    global user_profile_agent, product_rec_agent, subsidy_decision_agent
    global inventory_agent, ab_engine
    product_repository = ProductRepository()
    user_profile_agent = UserProfileAgent(feature_store=feature_store)
    product_rec_agent = ProductRecAgent(product_repository=product_repository)
    subsidy_decision_agent = SubsidyDecisionAgent(
        product_repository=product_repository
    )
    inventory_agent = InventoryAgent()
    ab_engine = ab_engine_instance or ABTestEngine()

    graph = StateGraph(PipelineState)
    graph.add_node("init", init_node)
    graph.add_node("user_profile", user_profile_node)
    graph.add_node("parallel_product", parallel_product_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "user_profile")
    graph.add_edge("user_profile", "parallel_product")
    graph.add_edge("parallel_product", "inventory")
    graph.add_edge("inventory", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()
