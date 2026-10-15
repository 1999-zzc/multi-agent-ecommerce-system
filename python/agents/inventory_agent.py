"""库存决策 Agent：过滤缺货商品，并生成库存预警和限购策略。"""
from __future__ import annotations
from typing import Any
from models.schemas import InventoryResult
from .base_agent import BaseAgent

SAFETY_STOCK_THRESHOLD = 50
LOW_STOCK_THRESHOLD = 100
HOT_ITEM_PURCHASE_LIMIT = 2


class InventoryAgent(BaseAgent):
    """检查候选商品库存，并返回可售 ID、库存预警和限购策略。"""

    def __init__(self):
        from config import get_settings

        settings = get_settings()
        super().__init__(
            name="inventory",
            timeout=settings.agent_timeout_inventory,
        )

    async def _execute(self, **kwargs: Any) -> InventoryResult:
        products: list[Any] = kwargs.get("products", [])
        available_products: list[str] = []
        low_stock_alerts: list[dict[str, Any]] = []
        purchase_limits: dict[str, int] = {}
        for product in products:
            if product.stock <= 0:
                continue
            available_products.append(product.product_id)
            if product.stock <= SAFETY_STOCK_THRESHOLD:
                low_stock_alerts.append(
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "current_stock": product.stock,
                        "level": "critical",
                        "action": "urgent_restock",
                    }
                )
            elif product.stock <= LOW_STOCK_THRESHOLD:
                low_stock_alerts.append(
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "current_stock": product.stock,
                        "level": "warning",
                        "action": "plan_restock",
                    }
                )
            limit = self._calc_purchase_limit(product)
            if limit is not None:
                purchase_limits[product.product_id] = limit
        return InventoryResult(
            success=True,
            available_products=available_products,
            low_stock_alerts=low_stock_alerts,
            purchase_limits=purchase_limits,
            data={
                "total_checked": len(products),
                "available_count": len(available_products),
                "alert_count": len(low_stock_alerts),
            },
            confidence=0.95,
        )

    def _calc_purchase_limit(self, product: Any) -> int | None:
        """根据库存深度和热门标签计算限购数量。"""
        is_hot = "新品" in product.tags or "旗舰" in product.tags
        if product.stock <= SAFETY_STOCK_THRESHOLD:
            return 1
        if product.stock <= LOW_STOCK_THRESHOLD and is_hot:
            return HOT_ITEM_PURCHASE_LIMIT
        if is_hot and product.stock <= 300:
            return 3
        return None
