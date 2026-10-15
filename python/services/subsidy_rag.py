from __future__ import annotations
import json
from datetime import date
from pathlib import Path
from typing import Any
from models.schemas import Product


class SubsidyRAG:
    """从本地政策知识库召回与候选商品、用户地区相关的政策片段。"""

    def __init__(self, knowledge_path: str | Path | None = None):
        default_path = Path(__file__).resolve().parent.parent / "knowledge" / "subsidy_policies.json"
        self.knowledge_path = Path(knowledge_path) if knowledge_path else default_path
        payload = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        self.policies: list[dict[str, Any]] = payload.get("policies", [])

    def retrieve(
        self,
        products: list[Product],
        context: dict[str, Any],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """按有效期、商品类目和地区关键词召回最相关的政策。"""
        today = date.today().isoformat()
        product_text = " ".join(
            f"{product.name} {product.category} {' '.join(product.tags)}"
            for product in products
        )
        context_text = json.dumps(context, ensure_ascii=False)
        scored: list[tuple[int, dict[str, Any]]] = []
        for policy in self.policies:
            if not policy["effective_from"] <= today <= policy["effective_to"]:
                continue
            category_hits = sum(
                category in product_text for category in policy.get("categories", [])
            )
            region_hits = sum(
                region in context_text for region in policy.get("regions", [])
                if region != "全国"
            )
            is_national = "全国" in policy.get("regions", [])
            score = category_hits * 2 + region_hits + int(is_national)
            if category_hits:
                scored.append((score, policy))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [policy for _, policy in scored[:top_k]]
