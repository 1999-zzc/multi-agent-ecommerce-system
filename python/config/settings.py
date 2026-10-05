# BaseSettings 是 Pydantic 提供的配置基类。
# 它可以自动从环境变量、.env 文件中读取配置。
from pydantic_settings import BaseSettings

# lru_cache 用来缓存 get_settings() 的返回值,避免每次调用都重新读取配置。
from functools import lru_cache


class Settings(BaseSettings):
    """项目全局配置。字段名会自动对应 ECOM_ 开头的环境变量。"""

    # 应用基础信息。
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # LLM 配置。
    # llm_api_key 是大模型 API key;为空时,创建 ChatOpenAI 可能会报缺少凭证。
    llm_api_key: str = ""

    # OpenAI 兼容接口地址。这里默认指向 MiniMax 的兼容接口。
    llm_base_url: str = "https://api.minimax.chat/v1"

    # 使用的大模型名称。
    llm_model: str = "MiniMax-M1"

    # 默认温度和最大 token。部分 Agent 会在自己的文件里覆盖这些参数。
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # 千问 Embedding 配置。
    # qwen_api_key 为空时需要在 .env 里配置 ECOM_QWEN_API_KEY。
    qwen_api_key: str = ""

    # 千问 OpenAI 兼容接口地址。
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 千问文本向量模型。
    qwen_embedding_model: str = "text-embedding-v4"

    # text-embedding-v4 默认向量维度。Milvus collection 建表时要用同一个维度。
    embedding_dim: int = 1024

    # Redis 配置,用于实时特征 Feature Store。
    redis_url: str = "redis://localhost:6379/0"

    # 特征缓存过期时间,单位秒。2592000 秒 = 30 天，覆盖 RFM 的 30 天计算窗口。
    feature_ttl_seconds: int = 2592000

    # 业务数据库配置。ProductRepository 会用它连接真实商品数据库。
    database_url: str = "mysql+pymysql://root:change_me@localhost:3306/ecommerce"

    # Milvus 连接地址。本地 Milvus 默认是 http://localhost:19530。
    milvus_uri: str = "http://localhost:19530"

    # Milvus 中保存商品向量的 collection 名称。
    milvus_collection: str = "product_embeddings"

    # A/B 测试配置。
    ab_test_enabled: bool = True

    # 默认分桶数量。100 表示把用户稳定分到 0-99 的桶里。
    ab_test_default_bucket_count: int = 100

    # 各 Agent 超时时间配置,单位秒。
    # 当前 BaseAgent 保存了 timeout,后续可进一步接 asyncio.wait_for 强制超时。
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_subsidy_decision: float = 8.0
    agent_timeout_inventory: float = 5.0

    # Pydantic Settings 配置:
    # - env_file=".env": 自动读取 python/.env
    # - env_prefix="ECOM_": 环境变量统一使用 ECOM_ 前缀
    # 例如 llm_api_key 对应 ECOM_LLM_API_KEY。
    model_config = {"env_file": ".env", "env_prefix": "ECOM_"}


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置对象。加缓存后,整个进程中通常只会创建一次 Settings。"""
    return Settings()
