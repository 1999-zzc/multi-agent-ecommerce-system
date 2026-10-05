"""离线推荐评估指标。"""

from __future__ import annotations

import math


def recall_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算 Recall@K：用户真实感兴趣商品被前 K 个推荐命中的比例。"""
    if not relevant_ids or k <= 0:
        return 0.0
    hits = len(set(recommended_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """计算 NDCG@K：命中越靠前得分越高。
    前 K 个推荐结果里，相关商品排得越靠前，分数越高。"""
    if not relevant_ids or k <= 0:
        return 0.0
    # 实际推荐排序得分
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, product_id in enumerate(recommended_ids[:k])
        if product_id in relevant_ids
    )
    # 理想情况下推荐得分
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mrr(recommended_ids: list[str], relevant_ids: set[str]) -> float:
    """ 第一个推荐对的商品”排在第几名。越靠前，分数越高"""
    for rank, product_id in enumerate(recommended_ids, start=1):
        if product_id in relevant_ids:
            return 1 / rank
    return 0.0


def evaluate_rankings(
    samples: list[tuple[list[str], set[str]]],
    k: int = 10,
) -> dict[str, float]:
    """把多个用户的推荐结果放在一起做离线评测，先给每个用户算 Recall、NDCG、MRR，再分别求平均，得到整个推荐系统的总体效果"""
    if not samples:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "mrr": 0.0}

    recall_scores = [recall_at_k(recommended, relevant, k) for recommended, relevant in samples]
    ndcg_scores = [ndcg_at_k(recommended, relevant, k) for recommended, relevant in samples]
    mrr_scores = [mrr(recommended, relevant) for recommended, relevant in samples]
    return {
        f"recall@{k}": round(sum(recall_scores) / len(samples), 4),
        f"ndcg@{k}": round(sum(ndcg_scores) / len(samples), 4),
        "mrr": round(sum(mrr_scores) / len(samples), 4),
    }
