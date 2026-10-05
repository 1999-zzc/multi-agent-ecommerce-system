# 当前系统架构

## 1. 项目目标

项目为电商推荐请求同时提供两类结果：

- 普通个性化推荐商品；
- 根据用户地区和政策规则计算的补贴商品。

系统还会在最终返回前统一检查库存，避免把缺货商品返回给用户。

## 2. 请求链路

```text
FastAPI /api/v1/recommend
            |
            v
     UserProfileAgent
  Redis 行为 / context -> UserProfile
            |
            v
       asyncio.gather
       /               \
      v                 v
ProductRecAgent   SubsidyDecisionAgent
多路召回 + 精排     标签筛选 + 政策 RAG
      \                 /
       \               /
            v
       InventoryAgent
       统一库存校验
            |
            v
       Supervisor 聚合
  subsidy_products + products
```

画像必须先完成，因为普通推荐需要画像进行规则召回和 LLM 精排。补贴决策只依赖请求上下文、MySQL 商品标签和政策知识库，因此它和普通商品推荐在同一个 `asyncio.gather()` 中并行执行。两个分支完成后，再由 `InventoryAgent` 统一检查库存。

## 3. Agent 职责

### UserProfileAgent

从 `FeatureStore` 读取 Redis 中的近期行为；如果没有实时行为，就使用 FastAPI 的 `context`。随后调用结构化 LLM，并用 Pydantic 生成 `UserProfile`。

### ProductRecAgent

从真实 products 表读取商品详情，组合三路召回：

1. `VectorRecall`：Qwen Embedding + Milvus；
2. `HotRecall`：热度、库存和新品规则；
3. `RuleRecall`：画像偏好类目和价格区间规则。

三路结果按权重融合、去重，再根据用户画像调用 LLM 精排。Milvus 只返回 `product_id`，商品价格、库存和状态仍然回 MySQL 查询。

### SubsidyDecisionAgent

先调用 `ProductRepository.list_discount_products()`，从 MySQL 的 `products.tags` 中找带有补贴或折扣标签的在线有货商品。然后由 `SubsidyRAG` 根据政策有效期、商品类目和地区召回政策片段。

代码计算每个商品的补贴金额、补贴后价格以及两个归一化得分：

```text
price_score    = 补贴后价格越低，得分越高
discount_score = 补贴金额越大，得分越高
综合分         = 0.5 * price_score + 0.5 * discount_score
```

LLM 读取 `context` 中的 `province`、`city` 或 `region`，判断地方政策是否适用于用户，再返回最多 4 个 `plan_id`。代码会再次校验方案是否存在、地区是否匹配、商品是否重复，并按综合分排序。

### InventoryAgent

它不需要 LLM，只执行确定性库存规则：

- `stock <= 0` 的商品不进入最终结果；
- `stock <= 50` 生成紧急补货预警并限购 1 件；
- `stock <= 100` 生成普通补货预警；
- 新品或旗舰商品根据库存深度生成限购数量。

### BaseAgent

所有 Agent 继承 `BaseAgent`，统一处理调用计数、超时、指数退避重试、日志、耗时和失败 fallback。业务 Agent 只负责自己的 `_execute()`。

## 4. 数据存储职责

| 组件 | 作用 |
|---|---|
| MySQL | 商品权威数据：价格、库存、状态、标签和描述 |
| Redis | 用户近期行为和实时特征 |
| Milvus | 商品向量和语义召回，只返回商品 ID |
| 本地 JSON RAG | 国家和地方补贴政策知识片段 |


## 5. FastAPI 响应

`POST /api/v1/recommend` 返回：

```json
{
  "request_id": "...",
  "user_id": "U001",
  "subsidy_products": [],
  "products": [],
  "experiment_group": "control",
  "agent_results": {},
  "total_latency_ms": 0.0,
  "timestamp": "..."
}
```

补贴商品在前，普通推荐商品在后；两个列表会按 `product_id` 去重。

## 6. 政策知识库

知识库文件为 [`python/knowledge/subsidy_policies.json`](../python/knowledge/subsidy_policies.json)，当前示例规则参考：

- [国家发展改革委 2026 年消费品以旧换新通知](https://www.ndrc.gov.cn/xxgk/zcfb/tz/202512/t20251230_1402851.html)；
- [湖北省 2026 年智能产品消费补贴通知](https://swt.hubei.gov.cn/zfxxgk/zc/qtzdgkwj/202609/t20260930_6026747.shtml)。

知识库是项目演示用的政策快照，正式业务需要定期同步政策并在结算侧再次校验资格。
