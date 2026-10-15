"""
商品推荐Agent
- 召回层：千问 Embedding + Milvus 向量召回 + 商品仓储层查详情
- 排序层：LLM重排 + 特征交叉(用户画像 x 商品属性)
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations
import json
import time
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import get_settings
from models.schemas import Product, ProductRecResult, UserProfile
from services.product_repository import ProductRepository
from services.milvus_product_vector_store import MilvusProductVectorStore
from services.qwen_embedding import QwenEmbeddingClient
from services.recall import HotRecall, RuleRecall, VectorRecall
from .base_agent import BaseAgent
RERANK_PROMPT = """你是电商推荐排序专家。根据用户画像和候选商品,重新排序并选出最优的{num_items}个商品。

用户画像:
{user_profile}

候选商品:
{candidates}

排序原则:
1. 用户偏好类目优先
2. 价格在用户可接受范围内
3. 保证类目多样性(相邻商品尽量不同类目)
4. 新品适当加权

请输出商品ID列表(JSON数组),按推荐优先级排序:
["product_id_1", "product_id_2", ...]

只输出JSON数组,不要其他内容。"""


class ProductRecAgent(BaseAgent):
    """商品推荐 Agent,负责候选召回和 LLM 重排。"""

    def __init__(self, product_repository: ProductRepository | None = None):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        self.product_repository = product_repository or ProductRepository()
        self.embedding_client = QwenEmbeddingClient(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_embedding_model,
        )
        self.vector_store = MilvusProductVectorStore(
            uri=settings.milvus_uri,
            collection_name=settings.milvus_collection,
            dimension=settings.embedding_dim,
        )
        self.vector_recall = VectorRecall(
            product_repository=self.product_repository,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )
        self.hot_recall = HotRecall(product_repository=self.product_repository)
        self.rule_recall = RuleRecall(product_repository=self.product_repository)

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)
        candidates: list[Product] | None = kwargs.get("candidates")
        use_llm_rerank: bool = kwargs.get("use_llm_rerank", True)
        if candidates is None:
            recall_result = await self.recall(
                limit=num_items * 3,
                user_profile=user_profile,
            )
            candidates = recall_result.products
        candidates = self._optimize_before_rerank(candidates)
        if use_llm_rerank:
            products = await self._rank_products(user_profile, candidates, num_items)
        else:
            products = candidates[:num_items]
        return ProductRecResult(
            success=True,
            products=products,
            recall_strategy="qwen_embedding+milvus+repository",
            stage="full",
            data={
                "candidate_count": len(candidates),
                "profile_used": user_profile is not None,
                "llm_rerank_used": use_llm_rerank,
            },
            confidence=0.8 if user_profile else 0.65,
        )

    async def recall(
        self,
        limit: int,
        user_profile: UserProfile | None = None,
    ) -> ProductRecResult:
        """带超时和重试的粗召回入口。"""
        start = time.perf_counter()
        try:
            result = await self._retry_operation(
                lambda: self._recall(limit=limit, user_profile=user_profile)
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as exc:
            return ProductRecResult(
                success=False,
                products=[],
                recall_strategy="recall_fallback",
                stage="recall",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
                confidence=0.0,
            )

    async def _recall(
        self,
        limit: int,
        user_profile: UserProfile | None = None,
    ) -> ProductRecResult:
        """
        粗召回候选商品。

        在线请求里只做查询:
        - 不构建 Milvus 索引
        - 不批量生成商品 embedding
        - 只从已有索引和商品库拿候选
        """
        vector_products = await self.vector_recall.recall(user_profile, limit)
        hot_products = await self.hot_recall.recall(user_profile, limit)
        rule_products = await self.rule_recall.recall(user_profile, limit)
        candidates = self._merge_candidates(
            (vector_products, 0.6),
            (hot_products, 0.3),
            (rule_products, 0.1),
            limit=limit,
        )
        return ProductRecResult(
            success=True,
            products=candidates,
            recall_strategy="vector+hot+rule",
            stage="recall",
            data={
                "candidate_count": len(candidates),
                "vector_count": len(vector_products),
                "hot_count": len(hot_products),
                "rule_count": len(rule_products),
            },
            confidence=0.75,
        )

    async def rerank(
        self,
        user_profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
        use_llm_rerank: bool = True,
    ) -> ProductRecResult:
        """带超时和重试的精排入口；失败时保持粗召回结果可用。"""
        start = time.perf_counter()
        try:
            result = await self._retry_operation(
                lambda: self._rerank_stage(
                    user_profile=user_profile,
                    candidates=candidates,
                    num_items=num_items,
                    use_llm_rerank=use_llm_rerank,
                )
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except Exception as exc:
            return ProductRecResult(
                success=False,
                products=candidates[:num_items],
                recall_strategy="rerank_fallback_to_recall",
                stage="rerank",
                error=str(exc),
                data={"candidate_count": len(candidates)},
                latency_ms=(time.perf_counter() - start) * 1000,
                confidence=0.2,
            )

    async def _rerank_stage(
        self,
        user_profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
        use_llm_rerank: bool = True,
    ) -> ProductRecResult:
        """
        根据用户画像对已有候选商品做精排。

        如果没有可用用户画像,会直接回退到粗召回顺序。
        如果 LLM 排序失败,也会回退到粗召回顺序。
        """
        candidates = self._optimize_before_rerank(candidates)
        if use_llm_rerank:
            products = await self._rank_products(user_profile, candidates, num_items)
        else:
            products = candidates[:num_items]
        confidence = 0.8 if user_profile else 0.6
        return ProductRecResult(
            success=True,
            products=products,
            recall_strategy="rerank_existing_candidates",
            stage="rerank",
            data={
                "candidate_count": len(candidates),
                "profile_used": user_profile is not None,
                "llm_rerank_used": use_llm_rerank,
            },
            confidence=confidence,
        )

    def _merge_candidates(
        self,
        *weighted_groups: tuple[list[Product], float],
        limit: int,
    ) -> list[Product]:
        """按召回来源权重融合候选商品。"""
        product_map: dict[str, Product] = {}
        scores: dict[str, float] = {}
        for products, weight in weighted_groups:
            for rank, product in enumerate(products):
                product_map[product.product_id] = product
                scores[product.product_id] = scores.get(product.product_id, 0.0) + weight / (rank + 1)
        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [product_map[pid] for pid in ranked_ids[:limit]]

    def _optimize_before_rerank(self, candidates: list[Product]) -> list[Product]:
        """
        LLM 精排前的轻量排序优化。

        这里不替代 LLM,只是避免候选列表过于单一:
        - 新品加权: 新品更靠前
        - 类目打散: 避免连续同类目
        - 卖家去重: 避免同商家占满候选
        """
        boosted = sorted(
            candidates,
            key=lambda p: (
                "新品" in p.tags,
                p.score,
                p.stock > 0,
            ),
            reverse=True,
        )
        result = []
        used_ids = set()
        seller_count: dict[str, int] = {}
        last_category = ""
        while len(result) < len(boosted):
            picked = None
            for product in boosted:
                if product.product_id in used_ids:
                    continue
                if seller_count.get(product.seller_id, 0) >= 2:
                    continue
                if product.category == last_category and len(result) < len(boosted) - 1:
                    continue
                picked = product
                break
            if not picked:
                picked = next(
                    (p for p in boosted if p.product_id not in used_ids),
                    None,
                )
            if not picked:
                break
            result.append(picked)
            used_ids.add(picked.product_id)
            seller_count[picked.seller_id] = seller_count.get(picked.seller_id, 0) + 1
            last_category = picked.category
        return result

    async def _rank_products(
        self,
        user_profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
    ) -> list[Product]:
        ranked_ids = await self._rerank(user_profile, candidates, num_items)
        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break
        return final_products[:num_items]

    async def _rerank(
        self, profile: UserProfile | None, candidates: list[Product], num_items: int
    ) -> list[str]:
        if not profile:
            return [p.product_id for p in candidates[:num_items]]
        if not profile.preferred_categories:
            return [p.product_id for p in candidates[:num_items]]
        profile_summary = {
            "segments": [s.value for s in profile.segments],
            "preferred_categories": profile.preferred_categories,
            "price_range": list(profile.price_range),
        }
        candidate_summary = [
            {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
            for p in candidates
        ]
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )
        messages = [
            SystemMessage(content="你是电商推荐排序专家。"),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            return [p.product_id for p in candidates[:num_items]]
