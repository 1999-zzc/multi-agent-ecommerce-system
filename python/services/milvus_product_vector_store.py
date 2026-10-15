from __future__ import annotations
from pymilvus import DataType
from pymilvus import MilvusClient
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
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="product_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        schema.add_field(
            field_name="category",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="brand",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="price",
            datatype=DataType.FLOAT,
        )
        schema.add_field(
            field_name="status",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_products(self, products: list[Product], vectors: list[list[float]]) -> None:
        self.ensure_collection()
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
        if not rows:
            return
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
        self.ensure_collection()
        search_kwargs = {}
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["product_id"],
            **search_kwargs,
        )
        hits = results[0] if results else []
        return [hit["entity"]["product_id"] for hit in hits]

    def delete_product(self, product_id: str) -> None:
        """从 Milvus 删除单个商品向量。"""
        self.ensure_collection()
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
