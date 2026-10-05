# services 包的统一出口。
# 这里放的是非 Agent 的支撑能力: A/B 实验、特征存储、指标收集。
from .ab_test import ABTestEngine
from .feature_store import FeatureStore
from .metrics import MetricsCollector
from .product_event import ProductEvent
from .product_index_builder import ProductIndexBuilder
from .product_index_sync import ProductIndexSync
from .product_repository import ProductRepository
from .recommendation_evaluator import evaluate_rankings, mrr, ndcg_at_k, recall_at_k
from .subsidy_rag import SubsidyRAG

# 控制 `from services import *` 时导出的服务类。
__all__ = [
    "ABTestEngine",
    "FeatureStore",
    "MetricsCollector",
    "ProductEvent",
    "ProductIndexBuilder",
    "ProductIndexSync",
    "ProductRepository",
    "SubsidyRAG",
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "evaluate_rankings",
]
