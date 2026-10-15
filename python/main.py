"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - 获取个性化推荐
  POST /api/v1/recommend/graph    - 通过LangGraph pipeline推荐
  POST /api/v1/behavior           - 写入用户实时行为
  POST /api/v1/recommendations/feedback - 记录推荐后的点击/加购/购买
  GET  /api/v1/experiments        - 查看A/B实验状态
  GET  /api/v1/metrics            - 查看系统监控指标
  GET  /health                    - 健康检查
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from contextlib import asynccontextmanager
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from config import get_settings
from models.schemas import (

    BehaviorEventRequest,
    RecommendationFeedbackRequest,
    RecommendationRequest,
    RecommendationResponse,
)
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services.ab_test import ABTestEngine
from services.metrics import MetricsCollector
from services.feature_store import FeatureStore

logger = structlog.get_logger()
settings = get_settings()
ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
feature_store = FeatureStore(
    redis_client=redis_client,
    ttl=settings.feature_ttl_seconds,
)
supervisor = None
rec_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期函数。

    启动时:
    - 构建 LangGraph 推荐图

    关闭时:
    - 记录关闭日志
    """
    global supervisor, rec_graph
    supervisor = SupervisorOrchestrator(
        ab_engine=ab_engine,
        feature_store=feature_store,
    )
    rec_graph = build_recommendation_graph(
        feature_store=feature_store,
        ab_engine_instance=ab_engine,
    )
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")
    await redis_client.aclose()
app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 补贴决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """健康检查接口,用于确认服务是否启动。"""
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    if not supervisor:
        raise HTTPException(status_code=503, detail="Supervisor not initialized")
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    product_ids = list(
        dict.fromkeys(
            [product.product_id for product in response.subsidy_products]
            + [product.product_id for product in response.products]
        )
    )
    metrics_collector.record_recommendation_exposure(
        request_id=response.request_id,
        user_id=response.user_id,
        experiment_group=response.experiment_group,
        product_ids=product_ids,
    )
    for _ in product_ids:
        ab_engine.record_metric(
            "rec_strategy",
            response.experiment_group,
            "exposure",
            1.0,
            response.user_id,
        )
    return response


@app.post("/api/v1/behavior")
async def record_behavior(request: BehaviorEventRequest):
    """记录浏览、点击、收藏、加购或购买等用户实时行为。"""
    stored = await feature_store.record_behavior(
        user_id=request.user_id,
        behavior_type=request.behavior_type.value,
        item_id=request.item_id,
        metadata=request.metadata,
    )
    return {"success": stored}


@app.post("/api/v1/recommendations/feedback")
async def record_recommendation_feedback(request: RecommendationFeedbackRequest):
    """记录某次推荐曝光后的点击、加购或购买，用于计算线上推荐和 A/B 指标。"""
    event = metrics_collector.record_recommendation_event(
        request_id=request.request_id,
        user_id=request.user_id,
        event_type=request.event_type.value,
        product_id=request.product_id,
        amount=request.amount,
    )
    if not event:
        raise HTTPException(
            status_code=404,
            detail="recommendation request or product exposure not found",
        )
    group = event["experiment_group"]
    ab_engine.record_metric(
        "rec_strategy",
        group,
        request.event_type.value,
        1.0,
        request.user_id,
    )
    if request.event_type.value == "purchase":
        ab_engine.record_metric(
            "rec_strategy",
            group,
            "gmv",
            request.amount or 0.0,
            request.user_id,
        )
    metadata = {
        "source": "recommendation",
        "request_id": request.request_id,
        "experiment_group": group,
    }
    if request.amount is not None:
        metadata["amount"] = request.amount
    await feature_store.record_behavior(
        user_id=request.user_id,
        behavior_type=request.event_type.value,
        item_id=request.product_id,
        metadata=metadata,
    )
    return {"success": True, "experiment_group": group}


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """使用LangGraph状态图进行推荐 (展示LangGraph能力)"""
    if not rec_graph:
        return {"error": "Graph not initialized"}
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    result = await rec_graph.ainvoke(state)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "subsidy_products": [
            product.model_dump()
            for product in result.get("subsidy_products", [])
        ],
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "experiment_group": result.get("experiment_group", "control"),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.get("/api/v1/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@app.get("/api/v1/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}


def _collect_metrics(response: RecommendationResponse):
    """从推荐响应中提取各 Agent 执行结果,写入 MetricsCollector。"""
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
            error=result.error or "",
        )
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
