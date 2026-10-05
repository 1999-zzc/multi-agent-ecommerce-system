from __future__ import annotations

# DataType 用来定义 Milvus collection 的字段类型。
from pymilvus import DataType

# MilvusClient 是 pymilvus 新版高层客户端,封装建表、写入、搜索等操作。
from pymilvus import MilvusClient

# Product 是项目里的商品结构。
from models.schemas import Product


class MilvusProductVectorStore:
    """
    商品向量库。

    Milvus 只保存用于向量检索的字段:
    - product_id: 商品 ID
    - vector: 商品 embedding
    - category/brand/price/status: 检索过滤和排序需要的元数据

    商品名称、价格、库存等完整信息仍然从 ProductRepository 查询。
    """

    def __init__(self, uri: str, collection_name: str, dimension: int):
        # 连接 Milvus 服务。
        self.client = MilvusClient(uri=uri)

        # 保存 collection 名称。
        self.collection_name = collection_name

        # 保存向量维度,必须和 embedding 模型输出维度一致。
        self.dimension = dimension

    def ensure_collection(self) -> None:
        # 如果 collection 已存在,不重复创建。
        if self.client.has_collection(self.collection_name):
            return

        # 创建 collection schema。
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        # product_id 作为主键,用来和业务数据库里的商品 ID 对齐。
        schema.add_field(
            field_name="product_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )

        # vector 保存商品语义向量。
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )

        # category 保存类目,后续可以扩展为 Milvus 过滤条件。
        schema.add_field(
            field_name="category",
            datatype=DataType.VARCHAR,
            max_length=64,
        )

        # brand 保存品牌,后续可做品牌过滤或品牌多样性。
        schema.add_field(
            field_name="brand",
            datatype=DataType.VARCHAR,
            max_length=64,
        )

        # price 保存价格,支持类似 price >= 5000 and price <= 8000 的过滤。
        schema.add_field(
            field_name="price",
            datatype=DataType.FLOAT,
        )

        # status 保存商品状态,只召回 online 商品。
        schema.add_field(
            field_name="status",
            datatype=DataType.VARCHAR,
            max_length=32,
        )

        # 创建向量索引参数。
        index_params = self.client.prepare_index_params()

        # AUTOINDEX 让 Milvus 自动选择合适索引;COSINE 适合文本语义相似度。
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # 真正创建 collection。
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_products(self, products: list[Product], vectors: list[list[float]]) -> None:
        # 先确保 collection 存在。
        self.ensure_collection()

        # 把 Product 和 embedding 组装成 Milvus 行数据。
        rows = [
            {
                "product_id": product.product_id,
                "vector": vector,
                "category": product.category,
                "brand": product.brand,
                "price": product.price,
                "status": "online",
            }
            for product, vector in zip(products, vectors)
        ]

        # 没有数据时直接返回。
        if not rows:
            return

        # upsert 表示有就更新,没有就插入。
        self.client.upsert(
            collection_name=self.collection_name,
            data=rows,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int,
        filter_expr: str = "",
    ) -> list[str]:
        # 先确保 collection 存在。
        self.ensure_collection()

        # Milvus search 支持 filter 表达式。为空时表示不过滤。
        search_kwargs = {}
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        # 用 query 向量在 Milvus 中搜索最相似的商品向量。
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["product_id"],
            **search_kwargs,
        )

        # Milvus 返回二维结果: 第一层是 query,第二层是命中的商品。
        hits = results[0] if results else []

        # 提取 product_id,交给 ProductRepository 查询完整商品信息。Milvus 搜索返回的是一组 hit，
        # 我这里只保留每条 hit 对应的商品 ID，方便后面拿这些 product_id 去 MySQL 查询完整商品信息。
        return [hit["entity"]["product_id"] for hit in hits]

    def delete_product(self, product_id: str) -> None:
        """从 Milvus 删除单个商品向量。"""
        # 删除前确保 collection 存在。
        self.ensure_collection()

        # Milvus 使用表达式删除数据。
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'product_id == "{product_id}"',
        )


"""
1.建立表collection_name，添加表元素，product_id作为主键，vector作为查询向量，另外还添加了brand，price，status，category
2.通过zip把product和vector绑定在一起，然后插入milvus
3.搜索+filter过滤，查询到后返回produc_id组成一个新的列表并交给MySQL查询完整商品信息
4.按照product_id删除数据
"""

"""
name、category、brand、price、tags,虽然也参与了 embedding 文本构造，但仍然作为标量字段单独存入 Milvus。
前者用于增强商品向量的语义表达，后者用于精确过滤，因此属于有目的的数据冗余，而不是无意义的重复存储。
"""