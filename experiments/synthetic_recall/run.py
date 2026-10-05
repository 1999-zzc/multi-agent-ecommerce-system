"""Reproducible synthetic ranking experiment; no databases or model APIs."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20260929
DIMENSION = 1024


@dataclass
class Product:
    product_id: str
    category: str
    brand: str
    price: float
    tags: list[str]
    seller_id: str
    stock: int
    score: float = 0.0


def load_project_functions():
    """Load only pure methods, avoiding Agent constructors and external clients."""
    source = ROOT / "python/agents/product_rec_agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "ProductRecAgent")
    names = {"_merge_candidates", "_optimize_before_rerank"}
    methods = [node for node in agent.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    namespace = {"Product": Product}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    evaluator = runpy.run_path(str(ROOT / "python/services/recommendation_evaluator.py"))
    return namespace, evaluator


def embed(product):
    """Sparse hashed attributes in a 1024-coordinate space, NOT Qwen embeddings."""
    tokens = [(f"category:{product.category}", 2.0), (f"brand:{product.brand}", 0.5)]
    tokens += [(f"tag:{tag}", 0.7) for tag in product.tags]
    result = Counter()
    for token, weight in tokens:
        position = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % DIMENSION
        result[position] += weight
    return normalize(result)


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def build_dataset():
    rng = random.Random(SEED)
    products, appeal = [], {}
    for index in range(2000):
        pid = f"P{index + 1:05d}"
        tags = rng.sample([f"style_{i}" for i in range(12)], 2)
        if rng.random() < 0.1:
            tags.append("新品")
        products.append(Product(pid, f"C{index % 20:02d}", f"B{rng.randrange(30):02d}",
                                round(math.exp(rng.uniform(math.log(50), math.log(8000))), 2),
                                tags, f"S{rng.randrange(100):03d}", rng.randint(1, 500)))
        appeal[pid] = min(rng.paretovariate(1.6), 20.0)

    events = []
    for index in range(200):
        uid = f"U{index + 1:03d}"
        categories = rng.sample([f"C{i:02d}" for i in range(20)], 2)
        styles = set(rng.sample([f"style_{i}" for i in range(12)], 2))
        budget = math.exp(rng.uniform(math.log(100), math.log(6000)))
        # Hidden preferences generate interactions; rankers only see training events.
        weights = [math.exp(
            2.0 * (p.category == categories[0]) + 1.0 * (p.category == categories[1])
            + 0.6 * len(styles.intersection(p.tags))
            - 0.7 * abs(math.log(p.price / budget)) + 0.4 * math.log(appeal[p.product_id])
        ) for p in products]
        chosen = set()
        while len(chosen) < 5:
            position = rng.choices(range(len(products)), weights=weights, k=1)[0]
            if position in chosen:
                continue
            chosen.add(position)
            ordinal = len(chosen) - 1
            split = "train" if ordinal < 4 else ("validation" if index < 100 else "test")
            day = 1 + ordinal * 10 if split == "train" else (49 if split == "validation" else 55)
            events.append({"user_id": uid, "product_id": products[position].product_id,
                           "day": day + rng.random(), "split": split})
    events.sort(key=lambda event: event["day"])
    return products, events


def mean_metrics(rows, evaluator):
    result = {}
    for k in (10, 20):
        values = [evaluator["recall_at_k"](row[0], row[1], k) for row in rows]
        result[f"recall@{k}"] = statistics.mean(values)
        result[f"hits@{k}"] = sum(value > 0 for value in values)
    result["ndcg@10"] = statistics.mean(evaluator["ndcg_at_k"](ids, rel, 10) for ids, rel in rows)
    result["mrr@10"] = statistics.mean(evaluator["mrr"](ids[:10], rel) for ids, rel in rows)
    return result


def paired_bootstrap(deltas):
    rng = random.Random(SEED + 1)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return [means[49], means[1949]]


def main():
    methods, evaluator = load_project_functions()
    products, events = build_dataset()
    catalog = {product.product_id: product for product in products}
    train = [event for event in events if event["split"] == "train"]
    validation = [event for event in events if event["split"] == "validation"]
    test = [event for event in events if event["split"] == "test"]
    assert len(train) == 800 and len(validation) == len(test) == 100
    assert max(e["day"] for e in train) < min(e["day"] for e in validation)
    assert max(e["day"] for e in validation) < min(e["day"] for e in test)

    histories = {f"U{i + 1:03d}": [] for i in range(200)}
    counts = Counter(event["product_id"] for event in train)
    for event in train:
        histories[event["user_id"]].append(catalog[event["product_id"]])
    for product in products:
        product.score = float(counts[product.product_id])
    vectors = {p.product_id: embed(p) for p in products}

    names = ["Popular proxy", "Rule proxy", "Vector proxy", "Weighted Fusion component",
             "Fusion then random sort", "Fusion then project rule rerank"]
    samples = {name: [] for name in names}
    details = []
    rng = random.Random(SEED + 2)
    shuffle_rng = random.Random(SEED + 3)
    for event in test:
        uid, positive = event["user_id"], event["product_id"]
        history = histories[uid]
        seen = {p.product_id for p in history}
        pool = [pid for pid in catalog if pid not in seen and pid != positive]
        ids = rng.sample(pool, 99) + [positive]
        rng.shuffle(ids)
        assert len(set(ids)) == 100 and positive in ids and not seen.intersection(ids)
        candidates = [catalog[pid] for pid in ids]

        profile = Counter()
        for product in history:
            profile.update(vectors[product.product_id])
        profile = normalize(profile)
        categories = Counter(p.category for p in history)
        budget = statistics.median(p.price for p in history)

        # Each proxy sees the same candidates and pre-cutoff history, never the label.
        vector = sorted(candidates, key=lambda p: sum(
            value * profile.get(coordinate, 0.0) for coordinate, value in vectors[p.product_id].items()
        ), reverse=True)[:30]
        popular = sorted(candidates, key=lambda p: p.score, reverse=True)[:30]
        rule = sorted(candidates, key=lambda p: (
            categories[p.category], -abs(math.log(p.price / budget))
        ), reverse=True)[:30]
        fusion = methods["_merge_candidates"](
            None, (vector, 0.6), (popular, 0.3), (rule, 0.1), limit=30
        )
        shuffled = sorted(fusion, key=lambda p: (p.stock > 0, shuffle_rng.random()), reverse=True)
        reranked = methods["_optimize_before_rerank"](None, fusion)
        rankings = {}
        for name, ranking in zip(names, [popular, rule, vector, fusion, shuffled, reranked]):
            ranked_ids = [p.product_id for p in ranking]
            assert len(ranked_ids) == len(set(ranked_ids)) == 30
            assert set(ranked_ids).issubset(ids)
            samples[name].append((ranked_ids, {positive}))
            rankings[name] = ranked_ids
        details.append({"user_id": uid, "day": event["day"], "positive_id": positive,
                        "candidate_ids": ids, "rankings": rankings})

    metrics = {name: mean_metrics(rows, evaluator) for name, rows in samples.items()}
    vector = metrics["Vector proxy"]
    fusion = metrics["Weighted Fusion component"]
    comparisons = {}
    for k in (10, 20):
        key = f"recall@{k}"
        deltas = [evaluator["recall_at_k"](b[0], b[1], k)
                  - evaluator["recall_at_k"](a[0], a[1], k)
                  for a, b in zip(samples["Vector proxy"], samples["Weighted Fusion component"])]
        comparisons[key] = {"absolute_delta": fusion[key] - vector[key],
                            "relative_change_percent": (fusion[key] / vector[key] - 1) * 100
                            if vector[key] else None,
                            "paired_bootstrap_95_percent_interval": paired_bootstrap(deltas)}

    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [ROOT / "python/agents/product_rec_agent.py",
                                 ROOT / "python/services/recommendation_evaluator.py"]}
    results = {"status": "EXECUTED_SYNTHETIC_COMPONENT_EXPERIMENT_NOT_MODEL_BENCHMARK",
               "seed": SEED, "products": 2000, "users": 200,
               "positive_interactions": 1000, "split_counts": [800, 100, 100],
               "test_users": len({e["user_id"] for e in test}), "test_samples": 100,
               "positives_per_sample": 1, "candidates_per_sample": 100, "route_limit": 30,
               "synthetic_hash_vector_coordinates": DIMENSION,
               "llm_executed": False, "qwen_executed": False, "milvus_executed": False,
               "mysql_executed": False, "redis_executed": False,
               "behavior_log_10000_generated": False, "validation_used_for_tuning": False,
               "project_source_sha256": fingerprints, "metrics": metrics,
               "fusion_vs_vector": comparisons}
    (OUT / "dataset.json").write_text(json.dumps(
        {"seed": SEED, "products": [asdict(p) for p in products], "interactions": events},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "rankings.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 合成数据离线评测：实际执行结果", "",
             "本报告由 run.py 自动生成。这是合成数据、模拟召回通道的组件实验，不是 Qwen、Milvus 或 LLM 的实测。", "",
             f"固定种子：{SEED}。商品 2,000，用户 200，合成正反馈 1,000；按时间划分 800/100/100。",
             "测试集为 100 名用户各一条样本；每条 1 个正例 + 99 个均匀采样的未观测商品。",
             "每路取前 30，融合权重沿用项目 0.6/0.3/0.1；未使用验证集调参，未根据结果更换种子。", "",
             "| 方案 | Recall@10 | Recall@20 | NDCG@10 | MRR@10 | Top10/Top20 命中样本 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, score in metrics.items():
        lines.append(f"| {name} | {score['recall@10']:.4f} | {score['recall@20']:.4f} | "
                     f"{score['ndcg@10']:.4f} | {score['mrr@10']:.4f} | "
                     f"{score['hits@10']}/{score['hits@20']} |")
    lines += ["", "## Fusion 对 Vector proxy 的比较", ""]
    for key, change in comparisons.items():
        lo, hi = change["paired_bootstrap_95_percent_interval"]
        lines.append(f"- {key}：绝对差 {change['absolute_delta'] * 100:+.2f} 个百分点，"
                     f"相对变化 {change['relative_change_percent']:+.2f}%；"
                     f"配对 bootstrap 差值区间 [{lo:.4f}, {hi:.4f}]。")
    lines += ["", "## 实验边界", "",
              "- 实际复用了项目的 _merge_candidates、_optimize_before_rerank，以及 Recall/NDCG/MRR 函数。",
              "- 召回通道是代理：历史点击热度、类目价格规则、1,024 坐标的稀疏属性哈希余弦相似度。哈希向量不代表 Qwen Embedding 质量。",
              "- Fusion component 保留融合顺序；随机排序行仅复现项目随后打乱顺序的影响。未运行完整 Agent/DAG。",
              "- 所有候选库存为正；本轮不验证库存过滤、缺货场景或依赖故障。",
              "- 每路候选相同，测试标签不传给排序函数；画像和热度只使用训练期交互。",
              "- 合成偏好规则与商品属性有关，可能偏好这些代理算法；均匀采样负例不等价于全库召回。",
              "- 只生成 1,000 条正反馈，并未生成此前规划的 10,000 条多类型行为日志。",
              "- 没有调用 LLM，所以没有 LLM Rerank 的 NDCG 或提升值；没有测线上 CTR、延迟或并发。",
              "- Bootstrap 仅描述该合成样本的抽样不确定性，不能证明真实用户效果。",
              "- 复跑：在项目目录执行 python3 experiments/synthetic_recall/run.py。",
              "- dataset.json 保留商品和时间切分交互，rankings.json 保留每条候选、标签及排名，results.json 保留未舍入指标与源文件哈希。", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
