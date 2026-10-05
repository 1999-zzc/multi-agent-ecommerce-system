# 代码阅读路线

这份文档按照一次推荐请求的实际执行顺序介绍代码，适合第一次阅读项目时使用。

## 1. 从请求模型开始

入口在 [`python/main.py`](../python/main.py)。FastAPI 接收：

```json
{
  "user_id": "U001",
  "scene": "homepage",
  "num_items": 10,
  "context": {
    "province": "湖北省",
    "city": "武汉市",
    "recent_views": ["手机", "耳机"]
  }
}
```

`user_id` 必填，其余字段有默认值。`context` 是灵活字典，既可以放画像行为，也可以放补贴判断所需的地区。

## 2. 先看 BaseAgent

路径：[`python/agents/base_agent.py`](../python/agents/base_agent.py)

外部调用统一使用：

```python
result = await agent.run(**kwargs)
```

执行顺序是：

```text
run()
  -> _retry_execute()
  -> asyncio.wait_for(_execute(), timeout)
  -> 成功：写入 latency_ms 并记录日志
  -> 失败：重试耗尽后返回 fallback AgentResult
```

`_execute()` 是抽象方法。用户画像、商品推荐、补贴决策和库存 Agent 分别实现自己的业务逻辑，重试、超时和错误结果不需要重复编写。

## 3. UserProfileAgent

路径：[`python/agents/user_profile_agent.py`](../python/agents/user_profile_agent.py)

它先执行 `_collect_behavior()`：

1. 如果注入了 `FeatureStore`，从 Redis 读取用户近期行为；
2. Redis 没有有效数据时，从请求 `context` 构造行为上下文；
3. 将行为交给 `structured_llm`；
4. 由 `UserProfileLLMOutput` 和 Pydantic 校验字段；
5. 通过 `_build_profile()` 转换为项目内部的 `UserProfile`。

如果结构化输出校验失败，Agent 返回默认画像，`success=True` 但 `confidence=0.35`。Supervisor 发现置信度低于 0.5 后，不使用这个画像做精排，而是保留基础推荐顺序。

## 4. Supervisor 的主流程

路径：[`python/orchestrator/supervisor.py`](../python/orchestrator/supervisor.py)

```text
1. 生成 request_id 和 A/B 分组
2. 执行 UserProfileAgent
3. ProductRecAgent || SubsidyDecisionAgent
4. InventoryAgent 校验两类商品库存
5. 去掉重复商品
6. 返回 subsidy_products 和 products
```

关键并行代码是：

```python
rec_result, subsidy_result = await asyncio.gather(
    self.product_rec_agent.run(...),
    self.subsidy_decision_agent.run(...),
)
```

补贴决策不依赖普通推荐结果，因此可以并行。库存 Agent 需要等待两个结果拿到商品列表，所以放在并行阶段之后。

## 5. ProductRecAgent

路径：[`python/agents/product_rec_agent.py`](../python/agents/product_rec_agent.py)

商品推荐 Agent 的内部顺序为：

```text
VectorRecall + HotRecall + RuleRecall
                |
                v
          Weighted Fusion
                |
                v
     类目打散 / 卖家去重 / 新品加权
                |
                v
          LLM Rerank
```

`ProductRepository` 查询 MySQL 商品详情。向量召回只拿 `product_id`，之后必须回表读取最新的价格、库存和状态。

## 6. SubsidyDecisionAgent

路径：[`python/agents/subsidy_decision_agent.py`](../python/agents/subsidy_decision_agent.py)

它的顺序更适合拆成四步：

1. `list_discount_products()` 从 MySQL 的 `tags` 中找补贴/折扣商品；
2. `SubsidyRAG.retrieve()` 召回有效政策；
3. 代码计算补贴金额、补贴后价格和 0.5/0.5 综合分；
4. LLM 根据 `context` 判断地区，返回最多 4 个方案 ID。

代码不会让 LLM 自己计算金额。这样可以避免模型随意编造价格，也方便面试时解释评分公式。

## 7. InventoryAgent

路径：[`python/agents/inventory_agent.py`](../python/agents/inventory_agent.py)

它不调用 LLM，只读取候选商品的 `stock` 字段。`stock <= 0` 的商品从两个结果列表中剔除，低库存商品会放入 `low_stock_alerts`，新品和旗舰商品会生成限购数量。

## 8. LangGraph 入口

路径：[`python/orchestrator/graph.py`](../python/orchestrator/graph.py)

LangGraph 复现同一条链路：

```text
init -> user_profile -> parallel_product -> inventory -> aggregate
```

`parallel_product` 节点内部用 `asyncio.gather()` 并行运行普通推荐和补贴决策。Supervisor 和 LangGraph 是两种编排入口，业务 Agent 逻辑保持一致。

## 9. 从哪里开始看

建议顺序：

1. `models/schemas.py`：先认识输入、输出和 AgentResult；
2. `agents/base_agent.py`：理解所有 Agent 的公共生命周期；
3. `orchestrator/supervisor.py`：理解整体调用顺序；
4. `agents/user_profile_agent.py`：理解 Redis、Pydantic 和画像；
5. `agents/product_rec_agent.py`：理解召回、融合和精排；
6. `agents/subsidy_decision_agent.py`：理解 RAG、补贴计算和地区判断；
7. `agents/inventory_agent.py`：理解最终库存校验。
