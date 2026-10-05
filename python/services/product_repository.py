from __future__ import annotations

# asyncio.to_thread 用来把同步 SQLAlchemy 查询放到工作线程，避免阻塞异步 Agent。
import asyncio

# JSON 用来描述数据库里的 tags 字段,例如 ["旗舰", "新品"]。
from sqlalchemy import JSON

# Column/Table/MetaData 用来声明 products 表结构。
from sqlalchemy import Column, Float, Integer, MetaData, String, Table

# create_engine 创建数据库连接引擎。
from sqlalchemy import create_engine

# cast/or_/select 用来构造商品查询条件。
from sqlalchemy import String as SQLString
from sqlalchemy import cast, or_, select

# Product 是项目里的商品数据结构。
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
        # 延迟导入配置,避免模块导入时就读取 .env。
        from config import get_settings

        # 没传 database_url 时,使用项目配置里的数据库地址。
        settings = get_settings()
        self.database_url = database_url or settings.database_url

        # 创建 SQLAlchemy engine。这里不建表,只连接数据库。
        self.engine = create_engine(self.database_url)

        # MetaData 是 SQLAlchemy 对表结构的容器。
        self.metadata = MetaData()

        # 声明真实数据库里应该存在的 products 表结构。
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
        # 构造查询语句。
        stmt = select(self.products_table).where(
            self.products_table.c.status == "online"
        )

        # 执行查询并转换成 Product 列表。
        return await asyncio.to_thread(self._fetch_products, stmt)

    async def get_by_ids(self, product_ids: list[str]) -> list[Product]:
        """
        根据商品 ID 批量查询商品详情。

        Milvus 只返回 product_id,完整商品信息仍然从数据库 products 表查询。
        """
        # 没有 ID 时直接返回空列表。
        if not product_ids:
            return []

        # 构造查询语句。
        stmt = select(self.products_table).where(
            self.products_table.c.product_id.in_(product_ids)
        )

        # 先查出商品。
        products = await asyncio.to_thread(self._fetch_products, stmt)

        # 建立 product_id 到 Product 的映射。
        product_map = {product.product_id: product for product in products}

        # 按 Milvus 返回的 ID 顺序组装商品列表。
        return [product_map[pid] for pid in product_ids if pid in product_map]

    async def get_by_id(self, product_id: str) -> Product | None:
        """根据商品 ID 查询单个商品,用于增量更新索引。"""
        # 复用批量查询逻辑,保持数据转换路径一致。
        products = await self.get_by_ids([product_id])

        # 找不到商品时返回 None。
        return products[0] if products else None

    async def list_discount_products(
        self,
        discount_tags: list[str],
        limit: int = 50,
    ) -> list[Product]:
        """从 MySQL 查询带补贴/折扣标签的在线有货商品。"""
        # tags 是 JSON 字段。转成字符串后用 LIKE，可以匹配“国补15%”这类扩展标签。
        tag_text = cast(self.products_table.c.tags, SQLString)
        tag_conditions = [tag_text.like(f"%{tag}%") for tag in discount_tags]

        # 只查在线、有库存且命中任一补贴标签的商品。
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
        # 打开数据库连接。
        with self.engine.connect() as conn:
            # 执行 SQL 并取出所有行。
            rows = conn.execute(stmt).fetchall()

        # 将每一行转换成 Product 对象。row._mapping把数据库行转换成类似字典
        return [self._row_to_product(row._mapping) for row in rows]

    def _row_to_product(self, row) -> Product:
        """把数据库查询结果转换成项目内部 Product 模型。"""
        # tags 在真实数据库里应该是 JSON 数组;如果为空就给空列表。
        tags = row.get("tags") or []

        # 兼容某些数据库把 JSON 读成字符串的情况。
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # 返回统一的 Product 业务对象。
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
