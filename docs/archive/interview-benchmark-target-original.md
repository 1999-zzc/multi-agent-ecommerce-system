# 面试口径与实验边界

本文保留在 `archive/`，用于记录项目当前可以如实描述的能力边界。它不提供未经执行的性能数字，也不把代码结构描述成线上结果。

## 一句话介绍

这是一个面向电商推荐场景的 Multi-Agent 系统：先根据 Redis 实时行为和请求上下文生成用户画像，再并行执行普通商品推荐和补贴决策，最后通过库存 Agent 过滤缺货商品并聚合返回。

## Agent 分工

| Agent | 输入 | 输出 |
|---|---|---|
| UserProfileAgent | user_id、Redis 行为、context | Pydantic UserProfile |
| ProductRecAgent | UserProfile、MySQL 商品、Milvus 向量 | 普通推荐商品 |
| SubsidyDecisionAgent | context、MySQL 补贴标签、政策 RAG | 最多 4 个补贴商品 |
| InventoryAgent | 两类候选商品 | 可售 ID、库存预警、限购策略 |

`MarketingCopyAgent` 已删除，不应在当前简历或面试回答中提及。

## 可直接回答的链路

```text
UserProfileAgent
        |
        v
ProductRecAgent || SubsidyDecisionAgent
        |
        v
InventoryAgent
        |
        v
subsidy_products + products
```

普通推荐和补贴决策通过 `asyncio.gather()` 并行。库存检查必须等待两个分支拿到商品后执行，因此它位于并行阶段之后。

## 补贴决策口径

补贴 Agent 先从 MySQL `products.tags` 查找带有 `国补`、`政府补贴`、`地方补贴`、`补贴` 或 `折扣` 标签的在线有货商品，再从本地 RAG 召回有效政策。

代码计算：

```text
补贴后价格 = 原价 - min(原价 * 补贴比例, 补贴上限)
综合分 = 0.5 * 低补贴后价格分 + 0.5 * 大补贴金额分
```

LLM 根据 `context` 中的省、市或地区判断地方政策是否适用。代码会再次校验地区、方案 ID 和商品重复，最多返回 4 个商品。

## 不应夸大的内容

- 没有真实用户流量时，不能说线上 CTR、CVR 或 GMV 提升；
- 没有实际压测时，不能填写平均延迟、P95 或并发吞吐数字；
- 没有真实 MySQL、Redis、Milvus 和模型联调时，不能说完成了生产部署；
- 政策 RAG 是项目内的知识库快照，实际补贴资格需要以当地活动平台和结算结果为准；
- 自动化测试通过，只能说明测试用例通过，不能直接等同于线上可用率。

## 推荐展示的代码入口

- [Supervisor](../../python/orchestrator/supervisor.py)
- [LangGraph](../../python/orchestrator/graph.py)
- [SubsidyDecisionAgent](../../python/agents/subsidy_decision_agent.py)
- [InventoryAgent](../../python/agents/inventory_agent.py)
- [补贴知识库](../../python/knowledge/subsidy_policies.json)
