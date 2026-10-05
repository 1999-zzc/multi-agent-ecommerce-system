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

# sys/os 用于把 python/ 目录加入模块搜索路径。
# 这样直接运行 `python main.py` 时,也能找到 agents/config/services 等本地包。
import sys
import os

# 将当前文件所在目录插到 sys.path 最前面。
sys.path.insert(0, os.path.dirname(__file__))

# asynccontextmanager 用于定义 FastAPI 的启动/关闭生命周期。
from contextlib import asynccontextmanager

# structlog 用于结构化日志。
import structlog

# uvicorn 是 ASGI 服务器,用于本地启动 FastAPI。
import uvicorn

# FastAPI 是 Web 框架。
from fastapi import FastAPI, HTTPException

# CORS 中间件允许浏览器跨域访问接口,便于前端或 Swagger 调试。
from fastapi.middleware.cors import CORSMiddleware

# Redis 异步客户端，用于记录和读取用户实时行为。
from redis.asyncio import Redis

# 读取项目配置。
from config import get_settings

# 推荐请求和响应模型,FastAPI 会用它做参数校验和 OpenAPI 文档生成。
from models.schemas import (
    BehaviorEventRequest,
    RecommendationFeedbackRequest,
    RecommendationRequest,
    RecommendationResponse,
)

# Supervisor 编排器,是生产推荐接口使用的主流程。
from orchestrator.supervisor import SupervisorOrchestrator

# LangGraph 版本推荐流程,用于展示图编排能力。
from orchestrator.graph import build_recommendation_graph

# A/B 测试引擎。
from services.ab_test import ABTestEngine

# 内存指标收集器。
from services.metrics import MetricsCollector

# FeatureStore 将 Redis 行为序列聚合成用户画像需要的实时特征。
from services.feature_store import FeatureStore

# 当前模块日志对象。
logger = structlog.get_logger()

# 全局配置对象。
settings = get_settings()


# 创建全局 A/B 引擎。接口和 Supervisor 共用同一个实例,保证实验状态一致。
ab_engine = ABTestEngine()

# 创建全局指标收集器。
metrics_collector = MetricsCollector()

# 创建 Redis 客户端。from_url 只创建连接对象，真正网络连接发生在首次读写行为时。
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)

# 复用同一个 FeatureStore，确保行为接口写入的数据能被用户画像 Agent 读取。
feature_store = FeatureStore(
    redis_client=redis_client,
    ttl=settings.feature_ttl_seconds,
)

# Supervisor 和 LangGraph 都在应用启动生命周期里构建。
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

    # 创建 Supervisor，并把 A/B 引擎和实时特征服务注入进去。
    supervisor = SupervisorOrchestrator(
        ab_engine=ab_engine,
        feature_store=feature_store,
    )

    # 构建图编排版本的推荐流程。
    rec_graph = build_recommendation_graph(
        feature_store=feature_store,
        ab_engine_instance=ab_engine,
    )

    # 记录启动日志。
    logger.info("app.startup", model=settings.llm_model)

    # yield 之前是启动逻辑,yield 之后是关闭逻辑。
    yield

    # 记录关闭日志。
    logger.info("app.shutdown")

    # 关闭 Redis 连接池，避免应用退出后遗留连接。
    await redis_client.aclose()


# 创建 FastAPI 应用实例。
app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description="用户画像Agent + 商品推荐Agent + 补贴决策Agent，并行+聚合模式",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件。
# allow_origins=["*"] 表示允许任意来源访问,适合 demo;生产环境应改成白名单。
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

    # 调用 Supervisor 完成整条推荐链路。
    response = await supervisor.recommend(request)

    # 将各 Agent 的调用结果写入内存指标。
    _collect_metrics(response)

    # 补贴区和普通推荐区都属于本次曝光；用 dict 去重并保留展示顺序。
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

    # 同步记录 A/B 实验的曝光事件，后续反馈会在相同实验组下记录点击、购买和 GMV。
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
    # 行为写入 Redis 后，下一次推荐可以从 FeatureStore 读到更新后的实时特征。
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

    # 只有此前确实曝光过的推荐商品才能计入指标，避免无关行为污染实验数据。
    if not event:
        raise HTTPException(
            status_code=404,
            detail="recommendation request or product exposure not found",
        )

    # 在同一实验组下记录转化行为，供 experiments 接口计算每组漏斗数据。
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

    # 推荐反馈既是评估数据，也是下一次画像需要的实时行为，因此同步写入 FeatureStore。
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
    # 如果生命周期还没初始化图,返回错误。
    if not rec_graph:
        return {"error": "Graph not initialized"}

    # LangGraph 使用 dict state,所以把 Pydantic 请求模型转成状态字典。
    state = {
        "user_id": request.user_id,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }

    # 异步执行图。
    result = await rec_graph.ainvoke(state)

    # 图返回的是状态字典,这里整理成接口友好的 JSON。
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
    # 将内存中的实验配置和统计信息整理成可读 JSON。
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
    # 返回 Agent 指标和业务事件指标。
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.post("/api/v1/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    # 记录某个实验组的一次成功/失败结果,用于 Thompson Sampling 后验更新。
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
    # 直接运行 `python main.py` 时启动开发服务器。
    # reload=True 会监听文件变化自动重启,适合本地开发。
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
