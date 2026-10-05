# 多 Agent 电商推荐与补贴决策系统

这是一个用于学习推荐系统、LLM Agent 和 LangGraph 编排的 Python 项目。系统根据用户画像生成普通商品推荐，同时从补贴政策知识库中检索规则，为用户所在地区选择可用的补贴商品。

## 当前架构

```text
FastAPI 推荐请求
        |
        v
UserProfileAgent
  Redis 实时行为 + 请求 context -> Pydantic 结构化用户画像
        |
        v
asyncio.gather
  |-----------------------------------------------|
  v                                               v
ProductRecAgent                           SubsidyDecisionAgent
Qwen Embedding + Milvus                   MySQL 补贴标签候选
Vector/Hot/Rule 多路召回                  本地政策 RAG
加权融合 + LLM 精排                       LLM 地区判断
库存 stock > 0 过滤                       0.5 价格分 + 0.5 优惠金额分
  |                                               |
  |----------------------|------------------------|
                         v
                  InventoryAgent
                   库存统一校验
                         |
                         v
                  Supervisor 聚合
        subsidy_products 在前，products 在后
```

当前包含四个业务 Agent：

1. `UserProfileAgent`：分析 Redis 实时行为或请求上下文，生成结构化用户画像。
2. `ProductRecAgent`：执行向量、热门和规则召回，融合后进行 LLM 精排。
3. `SubsidyDecisionAgent`：检索补贴政策，判断地区适用性，返回最多 4 个补贴方案。
4. `InventoryAgent`：统一校验普通商品和补贴商品的库存，生成预警和限购策略。

库存检查由 `InventoryAgent` 统一完成。项目只删除了 `MarketingCopyAgent`。

## 补贴 RAG

补贴知识库存放在：

```text
python/knowledge/subsidy_policies.json
```

当前收录的示例政策包括：

- 2026 年全国数码和智能产品购新补贴；
- 2026 年全国一级能效家电以旧换新补贴；
- 2026 年湖北省智能产品消费补贴。

`SubsidyRAG` 会先根据政策有效期、商品类目和地区关键词召回相关政策片段，再把片段交给 LLM。这里使用的是简单、可解释的关键词 RAG，没有额外引入政策向量库。

政策文件保存来源链接和有效期。实际业务应通过定时任务更新政策知识库，最终资格与金额仍以当地活动平台结算结果为准。

## 补贴商品筛选

`SubsidyDecisionAgent` 不扫描所有商品，而是调用 `ProductRepository.list_discount_products()`，从 MySQL `products.tags` 中查找以下标签：

```text
国补、政府补贴、地方补贴、补贴、折扣
```

推荐的标签示例：

```json
["国补", "新品"]
["国补", "一级能效"]
["湖北补贴", "智能家居"]
```

只有 `status = "online"` 且 `stock > 0` 的商品会进入补贴候选集。`折扣` 标签只用于进入候选池，真正套用国家或地方政策时还必须命中对应活动标签。

## 补贴排序

系统先根据政策计算补贴金额和补贴后价格，再做归一化评分：

```text
price_score     = 补贴后价格越低，得分越高
discount_score  = 补贴金额越大，得分越高
composite_score = 0.5 * price_score + 0.5 * discount_score
```

LLM 读取用户 `context`、政策片段和候选方案，判断国家或地方政策是否适用于用户所在地。代码随后校验方案 ID、地区和商品去重，并按综合分返回最多 4 个商品。

## API 输入

主接口：

```text
POST /api/v1/recommend
```

示例请求：

```json
{
  "user_id": "U001",
  "scene": "homepage",
  "num_items": 10,
  "context": {
    "province": "湖北省",
    "city": "武汉市",
    "recent_views": ["手机", "耳机"],
    "avg_order_amount": 3000
  }
}
```

其中只有 `user_id` 必填。要判断地方补贴，建议在 `context` 中明确传入 `province` 和 `city`；没有地区时只允许选择全国政策。

## API 输出

接口一次性返回两类商品，补贴商品字段在前：

```json
{
  "request_id": "9b3d...",
  "user_id": "U001",
  "subsidy_products": [
    {
      "product_id": "P002",
      "name": "某品牌平板",
      "category": "平板",
      "brand": "某品牌",
      "stock": 100,
      "tags": ["国补"],
      "image_url": "",
      "original_price": 3999.0,
      "subsidy_amount": 500.0,
      "subsidized_price": 3499.0,
      "discount_rate": 0.125,
      "price_score": 0.82,
      "discount_score": 1.0,
      "composite_score": 0.91,
      "policy_id": "national_digital_2026",
      "policy_name": "2026年数码和智能产品购新补贴",
      "applicable_region": "全国",
      "policy_source": "https://www.ndrc.gov.cn/..."
    }
  ],
  "products": [],
  "experiment_group": "treatment_llm",
  "agent_results": {},
  "total_latency_ms": 0.0,
  "timestamp": "2026-10-05T20:00:00"
}
```

补贴区和普通推荐区会按 `product_id` 去重。普通推荐会额外多取 4 个候选，尽量保证去重后仍能返回请求数量。

## 目录说明

```text
python/
├── main.py
├── agents/
│   ├── base_agent.py
│   ├── user_profile_agent.py
│   ├── product_rec_agent.py
│   ├── inventory_agent.py
│   └── subsidy_decision_agent.py
├── knowledge/
│   └── subsidy_policies.json
├── orchestrator/
│   ├── supervisor.py
│   └── graph.py
├── services/
│   ├── feature_store.py
│   ├── product_repository.py
│   ├── subsidy_rag.py
│   ├── milvus_product_vector_store.py
│   └── recall/
├── models/schemas.py
├── config/settings.py
└── tests/
```

## 配置与启动

```bash
cd python
cp .env.example .env
pip install -r requirements.txt
python main.py
```

核心环境变量包括：

```text
ECOM_LLM_API_KEY
ECOM_QWEN_API_KEY
ECOM_DATABASE_URL
ECOM_REDIS_URL
ECOM_MILVUS_URI
```

项目不会自动创建 MySQL `products` 表，也不会自动写入商品数据。Milvus 商品索引需要先通过 `python/scripts/build_product_index.py` 离线构建。

如使用 Docker Compose，需要先在项目根目录设置真实的环境变量：

```bash
export MYSQL_ROOT_PASSWORD='replace_with_a_local_password'
export ECOM_LLM_API_KEY='replace_with_llm_key'
export ECOM_QWEN_API_KEY='replace_with_qwen_key'
docker compose up --build
```

不要把真实 `.env` 文件或 API key 提交到 GitHub。

## 验证

```bash
python tests/test_recommendation_core.py
python tests/test_ab_test.py
```

更多面试和历史实验资料位于 [`docs/`](docs/) 与 [`experiments/`](experiments/)；其中旧基准报告描述的是改造前架构，不能直接当作当前补贴链路的实测结果。
