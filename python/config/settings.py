from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """项目全局配置。字段名会自动对应 ECOM_ 开头的环境变量。"""
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_seconds: int = 2592000
    database_url: str = "mysql+pymysql://root:change_me@localhost:3306/ecommerce"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "product_embeddings"
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_subsidy_decision: float = 8.0
    agent_timeout_inventory: float = 5.0
    model_config = {"env_file": ".env", "env_prefix": "ECOM_"}


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置对象。加缓存后,整个进程中通常只会创建一次 Settings。"""
    return Settings()
