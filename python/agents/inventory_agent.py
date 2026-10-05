"""库存决策 Agent：过滤缺货商品，并生成库存预警和限购策略。"""

from __future__ import annotations

# Any 用于同时接收普通推荐商品和补贴商品。
from typing import Any

# InventoryResult 是库存 Agent 的统一返回结构。
from models.schemas import InventoryResult

# 继承 BaseAgent，复用超时、重试、耗时统计和失败兜底。
from .base_agent import BaseAgent

# 安全库存线：小于等于 50 时需要紧急补货。
SAFETY_STOCK_THRESHOLD = 50

# 低库存线：小于等于 100 时需要计划补货。
LOW_STOCK_THRESHOLD = 100

# 热门商品库存偏低时的限购数量。
HOT_ITEM_PURCHASE_LIMIT = 2


class InventoryAgent(BaseAgent):
    """检查候选商品库存，并返回可售 ID、库存预警和限购策略。"""

    def __init__(self):
        # 延迟读取配置，避免模块导入阶段加载环境变量。
        from config import get_settings

        settings = get_settings()
        super().__init__(
            name="inventory",
            timeout=settings.agent_timeout_inventory,
        )

    async def _execute(self, **kwargs: Any) -> InventoryResult:
        # products 可以同时包含 Product 和 SubsidizedProduct；两者都有下面使用的字段。
        products: list[Any] = kwargs.get("products", [])
        available_products: list[str] = []
        low_stock_alerts: list[dict[str, Any]] = []
        purchase_limits: dict[str, int] = {}

        for product in products:
            # 库存小于等于 0 时不可售，不进入最终结果。
            if product.stock <= 0:
                continue

            available_products.append(product.product_id)

            # 根据库存深度生成补货预警。
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

            # 根据库存和商品标签决定是否限购。
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
