"""LangGraph：画像 -> {普通推荐 || 补贴决策} -> 聚合。"""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Any, TypedDict
from langgraph.graph import END, StateGraph
from agents import (

    InventoryAgent,
    ProductRecAgent,
    SubsidyDecisionAgent,
    UserProfileAgent,
)
from models.schemas import Product, SubsidizedProduct, UserProfile
from services.ab_test import ABTestEngine
from services.product_repository import ProductRepository


class PipelineState(TypedDict, total=False):
    """LangGraph 各节点之间传递的状态。"""
    request_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str
    use_llm_rerank: bool
    user_profile: UserProfile | None
    subsidy_products: list[SubsidizedProduct]
    recommended_products: list[Product]
    final_products: list[Product]
    agent_results: dict[str, Any]
    total_latency_ms: float
    _start_time: float
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
    candidates = list(
        {
            product.product_id: product
            for product in [*subsidy_products, *recommended_products]
        }.values()
    )
    result = await inventory_agent.run(products=candidates)
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
