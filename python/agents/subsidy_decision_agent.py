"""补贴决策 Agent：政策 RAG + 地区判断 + 补贴商品排序。"""

from __future__ import annotations

# json 用来把请求上下文、政策片段和候选方案传给 LLM。
import json

# Any 用于 BaseAgent 的动态入参和内部方案字典。
from typing import Any

# LangChain 消息对象负责组装结构化 LLM 请求。
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Pydantic 约束 LLM 只能返回方案 ID 列表。
from pydantic import BaseModel, Field, field_validator

# 项目配置、业务模型和支撑服务。
from config import get_settings
from models.schemas import Product, SubsidizedProduct, SubsidyDecisionResult
from services.product_repository import ProductRepository
from services.subsidy_rag import SubsidyRAG

# BaseAgent 提供统一超时、重试、日志和失败兜底。
from .base_agent import BaseAgent

# MySQL products.tags 命中任意一个标记后，商品才进入补贴候选集。
DISCOUNT_TAGS = ["国补", "政府补贴", "地方补贴", "补贴", "折扣"]

# LLM 只做地区适用性判断和方案选择，价格与补贴金额都由代码计算。
SYSTEM_PROMPT = """你是电商补贴合规决策助手。

请结合用户上下文中的省、市或地区，判断候选补贴方案能否在用户所在地使用：
1. regions 包含“全国”的方案可以使用。
2. 地方方案必须与上下文地区明确匹配；上下文没有地区时不能选择地方方案。
3. 每个商品最多选择一个方案；可用方案不少于 4 个时必须返回 4 个，否则全部返回。
4. 优先选择 composite_score 更高的方案，并按分数从高到低返回。
5. 只能返回输入中存在的 plan_id，不得编造政策、价格或商品。
"""


class SubsidyLLMOutput(BaseModel):
    """LLM 的最小结构化输出。"""

    selected_plan_ids: list[str] = Field(default_factory=list)

    @field_validator("selected_plan_ids")
    @classmethod
    def unique_and_limit(cls, value: list[str]) -> list[str]:
        """去重并限制最多四个方案，避免模型返回重复或过多 ID。"""
        return list(dict.fromkeys(value))[:4]


class SubsidyDecisionAgent(BaseAgent):
    """从补贴商品中选出地区可用且综合得分最高的四个方案。"""

    def __init__(
        self,
        product_repository: ProductRepository | None = None,
        subsidy_rag: SubsidyRAG | None = None,
    ):
        settings = get_settings()
        super().__init__(
            name="subsidy_decision",
            timeout=settings.agent_timeout_subsidy_decision,
        )

        # 复用项目现有的 OpenAI 兼容模型配置。
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=512,
        )
        self.structured_llm = self.llm.with_structured_output(SubsidyLLMOutput)

        # Repository 查 MySQL，RAG 查本地政策知识库。
        self.product_repository = product_repository or ProductRepository()
        self.subsidy_rag = subsidy_rag or SubsidyRAG()

    async def _execute(self, **kwargs: Any) -> SubsidyDecisionResult:
        # context 可包含 province/city/region，由 LLM 判断地方政策是否适用。
        context: dict[str, Any] = kwargs.get("context", {})
        limit: int = max(1, min(int(kwargs.get("limit", 4)), 4))

        # 第一步：从 MySQL 查询带补贴或折扣标签的在线有货商品。
        products = await self.product_repository.list_discount_products(
            DISCOUNT_TAGS,
            limit=50,
        )
        if not products:
            return SubsidyDecisionResult(
                success=True,
                data={"candidate_count": 0, "reason": "no_discount_tagged_products"},
                confidence=1.0,
            )

        # 第二步：从政策知识库召回与商品类目、地区上下文相关的政策片段。
        policies = self.subsidy_rag.retrieve(products, context)
        plans = self._build_plans(products, policies)
        self._score_plans(plans)
        if not plans:
            return SubsidyDecisionResult(
                success=True,
                retrieved_policy_ids=[p["policy_id"] for p in policies],
                data={"candidate_count": len(products), "reason": "no_policy_match"},
                confidence=1.0,
            )

        # 第三步：把检索到的政策和已算好的方案分数交给 LLM 做地区判断。
        prompt = {
            "user_context": context,
            "policies": policies,
            "candidate_plans": [self._plan_for_llm(plan) for plan in plans],
            "return_count": limit,
            "score_formula": "0.5 * price_score + 0.5 * discount_score",
        }
        decision = await self.structured_llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(prompt, ensure_ascii=False)),
            ]
        )

        # 第四步：只接受真实存在、地区可验证且商品不重复的方案。
        plan_map = {plan["plan_id"]: plan for plan in plans}
        selected: list[dict[str, Any]] = []
        used_product_ids: set[str] = set()
        for plan_id in decision.selected_plan_ids:
            plan = plan_map.get(plan_id)
            if not plan or plan["product_id"] in used_product_ids:
                continue
            if not self._region_matches(plan["regions"], context):
                continue
            selected.append(plan)
            used_product_ids.add(plan["product_id"])

        # 模型少返回了合法方案时，按同一综合分规则补齐，保证有足够候选时返回 4 个。
        for plan in sorted(
            plans,
            key=lambda item: item["composite_score"],
            reverse=True,
        ):
            if len(selected) >= limit:
                break
            if plan["product_id"] in used_product_ids:
                continue
            if not self._region_matches(plan["regions"], context):
                continue
            selected.append(plan)
            used_product_ids.add(plan["product_id"])

        # 最终仍按确定性的 0.5/0.5 综合分排序，防止模型把顺序返回错。
        selected.sort(key=lambda plan: plan["composite_score"], reverse=True)
        selected = selected[:limit]

        return SubsidyDecisionResult(
            success=True,
            products=[self._to_subsidized_product(plan) for plan in selected],
            retrieved_policy_ids=[p["policy_id"] for p in policies],
            data={
                "candidate_count": len(products),
                "plan_count": len(plans),
                "selected_count": len(selected),
                "price_weight": 0.5,
                "discount_weight": 0.5,
                "disclaimer": "补贴结果为政策知识库预估，最终资格和金额以当地活动平台结算为准。",
            },
            confidence=0.9 if selected else 0.6,
        )

    def _build_plans(
        self,
        products: list[Product],
        policies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """根据政策公式为每个商品生成可供 LLM 选择的补贴方案。"""
        plans: list[dict[str, Any]] = []
        for product in products:
            if product.price <= 0:
                continue
            product_text = f"{product.name} {product.category} {' '.join(product.tags)}"
            for policy in policies:
                if not any(category in product_text for category in policy["categories"]):
                    continue

                # “折扣”标签只能让商品进入候选池；真正套用政策还必须命中政策活动标签。
                campaign_tags = policy.get("campaign_tags", [])
                if campaign_tags and not any(
                    campaign_tag in product_tag
                    for campaign_tag in campaign_tags
                    for product_tag in product.tags
                ):
                    continue

                required_tags = policy.get("required_product_tags", [])
                if required_tags and not any(tag in product.tags for tag in required_tags):
                    continue

                max_price = policy.get("max_product_price")
                if max_price is not None and product.price > max_price:
                    continue

                subsidy_amount = min(
                    product.price * policy["subsidy_rate"],
                    policy["max_subsidy"],
                )
                plans.append(
                    {
                        "plan_id": f"{product.product_id}::{policy['policy_id']}",
                        "product": product,
                        "product_id": product.product_id,
                        "policy_id": policy["policy_id"],
                        "policy_name": policy["policy_name"],
                        "regions": policy["regions"],
                        "policy_source": policy.get("source", ""),
                        "original_price": round(product.price, 2),
                        "subsidy_amount": round(subsidy_amount, 2),
                        "subsidized_price": round(product.price - subsidy_amount, 2),
                        "discount_rate": round(subsidy_amount / product.price, 4),
                    }
                )
        return plans

    def _score_plans(self, plans: list[dict[str, Any]]) -> None:
        """计算低补贴后价格分、优惠金额分及各占 0.5 的综合分。"""
        if not plans:
            return

        prices = [plan["subsidized_price"] for plan in plans]
        discounts = [plan["subsidy_amount"] for plan in plans]
        min_price, max_price = min(prices), max(prices)
        min_discount, max_discount = min(discounts), max(discounts)

        for plan in plans:
            price_score = (
                1.0
                if min_price == max_price
                else (max_price - plan["subsidized_price"]) / (max_price - min_price)
            )
            discount_score = (
                1.0
                if min_discount == max_discount
                else (plan["subsidy_amount"] - min_discount) / (max_discount - min_discount)
            )
            plan["price_score"] = round(price_score, 4)
            plan["discount_score"] = round(discount_score, 4)
            plan["composite_score"] = round(
                0.5 * price_score + 0.5 * discount_score,
                4,
            )

    def _plan_for_llm(self, plan: dict[str, Any]) -> dict[str, Any]:
        """只把地区判断和排序需要的字段传给 LLM。"""
        return {
            key: plan[key]
            for key in (
                "plan_id",
                "product_id",
                "policy_id",
                "regions",
                "original_price",
                "subsidy_amount",
                "subsidized_price",
                "price_score",
                "discount_score",
                "composite_score",
            )
        }

    def _region_matches(self, regions: list[str], context: dict[str, Any]) -> bool:
        """对 LLM 的地区结论再做一次白名单校验。"""
        if "全国" in regions:
            return True
        context_text = json.dumps(context, ensure_ascii=False)
        return any(region in context_text for region in regions)

    def _to_subsidized_product(self, plan: dict[str, Any]) -> SubsidizedProduct:
        """把内部方案转换成接口返回模型。"""
        product = plan["product"]
        return SubsidizedProduct(
            product_id=product.product_id,
            name=product.name,
            category=product.category,
            brand=product.brand,
            stock=product.stock,
            tags=product.tags,
            image_url=product.image_url,
            original_price=plan["original_price"],
            subsidy_amount=plan["subsidy_amount"],
            subsidized_price=plan["subsidized_price"],
            discount_rate=plan["discount_rate"],
            price_score=plan["price_score"],
            discount_score=plan["discount_score"],
            composite_score=plan["composite_score"],
            policy_id=plan["policy_id"],
            policy_name=plan["policy_name"],
            applicable_region="、".join(plan["regions"]),
            policy_source=plan["policy_source"],
        )

    def _fallback(
        self,
        latency_ms: float,
        exc: Exception,
        **kwargs: Any,
    ) -> SubsidyDecisionResult:
        """依赖异常时不返回未经政策确认的补贴商品。"""
        return SubsidyDecisionResult(
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            products=[],
            confidence=0.0,
        )
