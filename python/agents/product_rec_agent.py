"""
商品推荐Agent
- 召回层：千问 Embedding + Milvus 向量召回 + 商品仓储层查详情
- 排序层：LLM重排 + 特征交叉(用户画像 x 商品属性)
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations

# json 用来把用户画像、候选商品传给 LLM,也用来解析 LLM 返回的商品 ID 数组。
import json

# time 用于统计粗召回和精排两个阶段各自的耗时。
import time

# Any 表示任意类型,用于接收 run() 传进来的动态参数。
from typing import Any

# LangChain 消息对象: 系统消息设定角色,用户消息提供本次排序任务。
from langchain_core.messages import HumanMessage, SystemMessage

# OpenAI 兼容聊天模型客户端,这里实际可接 MiniMax 等兼容接口。
from langchain_openai import ChatOpenAI

# 读取 LLM 和 Agent 超时等配置。
from config import get_settings

# ProductRecResult 是推荐 Agent 的返回结构,Product/UserProfile 是业务对象。
from models.schemas import Product, ProductRecResult, UserProfile

# ProductRepository 负责查询商品数据。
# 它会访问真实数据库中的 products 表。
from services.product_repository import ProductRepository

# MilvusProductVectorStore 负责商品向量写入和相似度搜索。
from services.milvus_product_vector_store import MilvusProductVectorStore

# QwenEmbeddingClient 负责调用千问 embedding 模型生成向量。
from services.qwen_embedding import QwenEmbeddingClient

# 多路召回器。
from services.recall import HotRecall, RuleRecall, VectorRecall

# 继承 BaseAgent,获得统一的 run、重试、日志和 fallback。
from .base_agent import BaseAgent

# LLM 重排提示词。
# 输入是用户画像和候选商品,输出要求是商品 ID 的 JSON 数组。
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
        # 读取模型配置和 Agent 超时配置。
        settings = get_settings()

        # 初始化父类,声明这个 Agent 的名字是 product_rec。
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )

        # 推荐重排需要较稳定的输出,所以 temperature 设置为 0.3。
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )

        # 商品仓储层。Agent 通过它查询商品,而不是直接依赖写死的商品列表。
        self.product_repository = product_repository or ProductRepository()

        # 千问 Embedding 客户端,用于商品文本和用户查询文本向量化。
        self.embedding_client = QwenEmbeddingClient(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model=settings.qwen_embedding_model,
        )

        # Milvus 商品向量库,用于语义召回商品 ID。
        self.vector_store = MilvusProductVectorStore(
            uri=settings.milvus_uri,
            collection_name=settings.milvus_collection,
            dimension=settings.embedding_dim,
        )

        # 向量召回: 千问 embedding + Milvus。
        self.vector_recall = VectorRecall(
            product_repository=self.product_repository,
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )

        # 热门召回: 按商品热度/库存/新品等字段补充候选。
        self.hot_recall = HotRecall(product_repository=self.product_repository)

        # 规则召回: 根据业务规则补充关联商品。
        self.rule_recall = RuleRecall(product_repository=self.product_repository)

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        # user_profile 可能为空。没有画像时会按基础召回顺序返回。
        user_profile: UserProfile | None = kwargs.get("user_profile")

        # num_items 是最终想要的商品数量,默认 10 个。
        num_items: int = kwargs.get("num_items", 10)

        # 外部可以传入 candidates,这样就能复用第一阶段粗召回结果,避免重复查商品。
        candidates: list[Product] | None = kwargs.get("candidates")

        # A/B 对照组可以关闭 LLM 精排，只保留确定性规则排序。
        use_llm_rerank: bool = kwargs.get("use_llm_rerank", True)

        # 没有传 candidates 时,run() 会自己先做粗召回,保持单独调用也能工作。
        if candidates is None:
            recall_result = await self.recall(
                limit=num_items * 3,
                user_profile=user_profile,
            )
            candidates = recall_result.products

        # 对候选商品做排序前优化。
        candidates = self._optimize_before_rerank(candidates)

        # 实验组调用 LLM 精排，对照组直接使用规则优化后的顺序。
        if use_llm_rerank:
            products = await self._rank_products(user_profile, candidates, num_items)
        else:
            products = candidates[:num_items]

        # 返回完整推荐结果。
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
            # 粗召回失败时返回空候选，由 Supervisor 的后续兜底逻辑处理。
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
        # 多路召回。每一路最多取 limit 个,最后统一合并去重。
        vector_products = await self.vector_recall.recall(user_profile, limit)
        hot_products = await self.hot_recall.recall(user_profile, limit)
        rule_products = await self.rule_recall.recall(user_profile, limit)

        # 多路召回加权融合,再按分数排序去重。
        candidates = self._merge_candidates(
            (vector_products, 0.6),
            (hot_products, 0.3),
            (rule_products, 0.1),
            limit=limit,
        )

        # 返回召回结果。
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
            # 精排失败不丢弃已有候选，直接回退到粗召回顺序。
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
        # LLM 精排前先做轻量排序优化:
        # - 新品加权
        # - 类目打散
        # - 卖家去重
        candidates = self._optimize_before_rerank(candidates)

        # 对照组可关闭 LLM，直接使用规则优化后的候选顺序；实验组再调用 LLM 精排。
        if use_llm_rerank:
            products = await self._rank_products(user_profile, candidates, num_items)
        else:
            products = candidates[:num_items]

        # 有画像时置信度高一些;没有画像时说明只能按粗召回顺序返回。
        confidence = 0.8 if user_profile else 0.6

        # 返回精排结果。
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

        # 越靠前的商品位置分越高,再乘以召回通道权重。
        for products, weight in weighted_groups:
            for rank, product in enumerate(products):
                product_map[product.product_id] = product
                scores[product.product_id] = scores.get(product.product_id, 0.0) + weight / (rank + 1)

        # 按融合分从高到低返回。
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
        # 先做新品加权和基础分排序。
        boosted = sorted(
            candidates,
            key=lambda p: (
                "新品" in p.tags,
                p.score,
                p.stock > 0,
            ),
            reverse=True,
        )

        # 再做类目打散和卖家去重。
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

            # 如果严格打散找不到商品,就放宽条件拿第一个未使用商品。
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
        # 根据用户画像让 LLM 重新排序,返回商品 ID 列表。
        ranked_ids = await self._rerank(user_profile, candidates, num_items)

        # 建立 product_id -> Product 的映射,方便根据 LLM 返回的 ID 找回完整商品对象。
        id_to_product = {p.product_id: p for p in candidates}

        # 按 LLM 给出的顺序组装最终商品列表。
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])

        # 如果 LLM 返回的 ID 不够、重复或有非法 ID,就用候选商品补齐。
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        # 返回最终排序后的商品对象列表。
        return final_products[:num_items]

    async def _rerank(
        self, profile: UserProfile | None, candidates: list[Product], num_items: int
    ) -> list[str]:
        # 没有用户画像时,无法做个性化重排,直接返回候选商品前 num_items 个。
        if not profile:
            return [p.product_id for p in candidates[:num_items]]

        # 如果画像是兜底画像,通常没有明确偏好类目。
        # 这种情况下不调用 LLM,直接保持粗召回顺序,避免把低质量画像传给模型。
        if not profile.preferred_categories:
            return [p.product_id for p in candidates[:num_items]]

        # 把用户画像压缩成 LLM 容易理解的 JSON。
        profile_summary = {
            "segments": [s.value for s in profile.segments],
            "preferred_categories": profile.preferred_categories,
            "price_range": list(profile.price_range),
        }

        # 把候选商品压缩成只含排序需要的字段,避免把无关信息都塞给模型。
        candidate_summary = [
            {"id": p.product_id, "name": p.name, "category": p.category, "price": p.price, "tags": p.tags}
            for p in candidates
        ]

        # 把用户画像和候选商品填入重排提示词。
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )

        # 组装消息并调用 LLM。
        messages = [
            SystemMessage(content="你是电商推荐排序专家。"),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)
        try:
            # 取出模型文本输出。
            raw = response.content.strip()

            # 兼容模型输出 ```json ... ``` 代码块的情况。
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            # 期望得到 ["P001", "P002"] 这样的 ID 数组。
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            # 如果模型输出格式不对,降级为原始候选顺序。
            return [p.product_id for p in candidates[:num_items]]
