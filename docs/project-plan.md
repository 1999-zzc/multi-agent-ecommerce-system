# 项目实施说明

## 1. 项目范围

本项目只维护 Python 版本，重点展示：

- FastAPI 接口；
- UserProfileAgent、ProductRecAgent、SubsidyDecisionAgent 和 InventoryAgent；
- Supervisor 与 LangGraph 两种编排入口；
- Redis 实时特征、MySQL 商品库、Milvus 向量召回；
- 本地补贴政策 RAG；
- Pydantic 结构化输出、超时、重试、降级和 A/B 分组。

营销文案 Agent 已从当前版本删除。库存 Agent 仍然保留，并在两个商品分支结束后统一校验库存。

## 2. 数据流设计

```text
请求
  -> 用户画像
  -> {普通推荐 || 补贴决策}
  -> 库存检查
  -> 去重和聚合
  -> 响应
```

### 阶段一：用户画像

从 Redis 读取用户近期行为，例如浏览、点击、收藏、加购和购买。如果 Redis 没有数据，就使用 FastAPI 请求中的 `context`。LLM 输出由 Pydantic 校验，再转换为 `UserProfile`。

### 阶段二：普通推荐与补贴决策并行

普通推荐使用三路召回：

- Qwen Embedding + Milvus 向量召回；
- 热门和库存规则召回；
- 用户偏好类目和价格区间规则召回。

补贴决策直接从 MySQL 的商品标签筛选候选商品，随后从 [`python/knowledge/subsidy_policies.json`](../python/knowledge/subsidy_policies.json) 召回有效政策。LLM 读取地区上下文，判断地方政策能否使用。

### 阶段三：库存检查

`InventoryAgent` 检查普通商品和补贴商品的 `stock`：

- 没货商品不返回；
- 低库存商品生成预警；
- 新品、旗舰商品生成限购策略。

## 3. 商品数据库约定

真实数据库需要有 `products` 表，至少包含：

```text
product_id, name, category, price, description,
brand, seller_id, stock, tags, score, image_url, status
```

补贴候选标签可以使用：

```json
["国补", "新品"]
["国补", "一级能效"]
["湖北补贴", "智能家居"]
```

代码只查询数据库，不创建表、不插入模拟商品。商品向量需要先通过 `python/scripts/build_product_index.py` 写入 Milvus，在线请求只做检索。

## 4. 补贴排序规则

对每个商品和政策组合计算：

```text
subsidy_amount  = min(price * subsidy_rate, max_subsidy)
subsidized_price = price - subsidy_amount
composite_score  = 0.5 * price_score + 0.5 * discount_score
```

其中 `price_score` 反映补贴后价格，`discount_score` 反映补贴金额。LLM 只返回候选方案 ID，代码会做 ID、地区、重复商品和最终排序校验。

## 5. 配置和部署边界

需要配置：

```text
ECOM_LLM_API_KEY
ECOM_QWEN_API_KEY
ECOM_DATABASE_URL
ECOM_REDIS_URL
ECOM_MILVUS_URI
```

项目不会自动启动 MySQL、Redis 或 Milvus。部署时应先准备真实服务和商品表，再构建 Milvus 索引，最后启动 FastAPI。

补贴政策会变化，本项目的 JSON 是学习和演示用快照。正式系统应增加政策同步、版本管理、审核和结算时二次校验。

## 6. 验证计划

当前已有：

- BaseAgent 超时返回测试；
- Redis FeatureStore 行为写入测试；
- 用户画像默认降级测试；
- 多路召回去重测试；
- InventoryAgent 缺货过滤测试；
- 推荐曝光与点击/购买归因测试；
- Recall@K、NDCG@K 和 MRR 计算测试；
- A/B 稳定分桶和统计测试。

运行方式：

```bash
cd python
python tests/test_recommendation_core.py
python tests/test_ab_test.py
```

没有真实用户流量时，不把离线指标写成线上 CTR、CVR 或 GMV 提升；性能和推荐效果数字也必须在固定环境实际测量后再写入简历。
