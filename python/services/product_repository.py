from __future__ import annotations
import asyncio
from sqlalchemy import JSON
from sqlalchemy import Column, Float, Integer, MetaData, String, Table
from sqlalchemy import create_engine
from sqlalchemy import String as SQLString
from sqlalchemy import cast, or_, select
from models.schemas import Product


class ProductRepository:
    """
    商品仓储层。

    这个类按真实数据库访问方式写:
    - 不在代码里写死商品
    - 不创建数据库
    - 不插入测试数据
    - 只负责查询真实 products 表

    真实链路:
    ProductRecAgent -> ProductRepository -> products 表 -> Product 对象
    """

    def __init__(self, database_url: str | None = None):
        from config import get_settings

        settings = get_settings()
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url)
        self.metadata = MetaData()
        self.products_table = Table(
            "products",
            self.metadata,
            Column("product_id", String(64), primary_key=True),
            Column("name", String(255), nullable=False),
            Column("category", String(64), nullable=False),
            Column("price", Float, nullable=False),
            Column("description", String, default=""),
            Column("brand", String(64), default=""),
            Column("seller_id", String(64), default=""),
            Column("stock", Integer, default=0),
            Column("tags", JSON, default=list),
            Column("score", Float, default=0.0),
            Column("image_url", String, default=""),
            Column("status", String(32), default="online"),
        )

    async def list_all(self) -> list[Product]:
        """
        查询所有在线商品。

        这个方法用于构建 Milvus 商品向量索引。
        """
        stmt = select(self.products_table).where(
            self.products_table.c.status == "online"
        )
        return await asyncio.to_thread(self._fetch_products, stmt)

    async def get_by_ids(self, product_ids: list[str]) -> list[Product]:
        """
        根据商品 ID 批量查询商品详情。

        Milvus 只返回 product_id,完整商品信息仍然从数据库 products 表查询。
        """
        if not product_ids:
            return []
        stmt = select(self.products_table).where(
            self.products_table.c.product_id.in_(product_ids)
        )
        products = await asyncio.to_thread(self._fetch_products, stmt)
        product_map = {product.product_id: product for product in products}
        return [product_map[pid] for pid in product_ids if pid in product_map]

    async def get_by_id(self, product_id: str) -> Product | None:
        """根据商品 ID 查询单个商品,用于增量更新索引。"""
        products = await self.get_by_ids([product_id])
        return products[0] if products else None

    async def list_discount_products(
        self,
        discount_tags: list[str],
        limit: int = 50,
    ) -> list[Product]:
        """从 MySQL 查询带补贴/折扣标签的在线有货商品。"""
        tag_text = cast(self.products_table.c.tags, SQLString)
        tag_conditions = [tag_text.like(f"%{tag}%") for tag in discount_tags]
        stmt = (
            select(self.products_table)
            .where(
                self.products_table.c.status == "online",
                self.products_table.c.stock > 0,
                or_(*tag_conditions),
            )
            .limit(limit)
        )
        return await asyncio.to_thread(self._fetch_products, stmt)

    def _fetch_products(self, stmt) -> list[Product]:
        """执行 SQLAlchemy 查询语句,并把数据库行转换成 Product。"""
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [self._row_to_product(row._mapping) for row in rows]

    def _row_to_product(self, row) -> Product:
        """把数据库查询结果转换成项目内部 Product 模型。"""
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return Product(
            product_id=row["product_id"],
            name=row["name"],
            category=row["category"],
            price=row["price"],
            description=row.get("description") or "",
            brand=row.get("brand") or "",
            seller_id=row.get("seller_id") or "",
            stock=row.get("stock") or 0,
            tags=tags,
            score=row.get("score") or 0.0,
            image_url=row.get("image_url") or "",
        )
"""
1.先读取数据库连接地址并创建 SQLAlchemy的engine连接数据库，声明products表结构。后续通过self.products_table构造SQL查询，再通过 engine 执行查询。
2.列出所有在线商品。
3.通过milvus返回的product_id去查询表，并建立product_id到Product的映射，之后按Milvus返回的ID顺序组装商品列表。
4.根据单个productd查询_i商品
5.把查询到的product组装成统一的Product业务对象
"""
