from __future__ import annotations

# json 用来读取本地补贴政策知识库。
import json

# date 用来过滤尚未生效或已经过期的政策。
from datetime import date

# Path 用来定位 knowledge/subsidy_policies.json。
from pathlib import Path

# Any 用于表示灵活的请求上下文和政策字典。
from typing import Any

# Product 提供待匹配商品的类目、名称和标签。
from models.schemas import Product


class SubsidyRAG:
    """从本地政策知识库召回与候选商品、用户地区相关的政策片段。"""

    def __init__(self, knowledge_path: str | Path | None = None):
        # 默认知识库和 python/ 目录一起部署，不依赖运行命令所在目录。
        default_path = Path(__file__).resolve().parent.parent / "knowledge" / "subsidy_policies.json"
        self.knowledge_path = Path(knowledge_path) if knowledge_path else default_path

        # 启动时只读取一次政策 JSON，后续请求直接在内存中检索。
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
            # 过期或尚未生效的政策不进入 LLM 上下文。
            if not policy["effective_from"] <= today <= policy["effective_to"]:
                continue

            # 类目命中是主要召回信号，地区命中作为额外加分。
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

        # 返回政策原文片段，后续由 Agent 拼入 LLM 提示词，这就是 RAG 的增强环节。
        scored.sort(key=lambda item: item[0], reverse=True)
        return [policy for _, policy in scored[:top_k]]
