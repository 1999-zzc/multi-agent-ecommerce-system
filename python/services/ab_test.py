"""
A/B测试引擎
- 流量分桶：用户ID哈希取模分桶
- 实验层：Agent级别 / 模型级别 / Prompt级别实验
- MAB算法：Thompson Sampling动态分配流量
- 指标收集：CTR / CVR / GMV / 停留时长
"""

from __future__ import annotations

# hashlib 用于把 user_id 稳定映射到一个桶,保证同一用户每次进同一实验组。
import hashlib

# time 用于记录指标上报时间戳。
import time

# dataclass 让实验配置类更简洁;field 用于安全地创建默认 dict。
from dataclasses import dataclass, field

# Any 用于实验配置和指标中的灵活字段。
from typing import Any

# numpy 用于计算统计值,也用于 Thompson Sampling 中的 beta 分布采样。
import numpy as np


@dataclass
class Experiment:
    """一个 A/B 实验定义。"""

    # 实验唯一 ID,例如 rec_strategy。
    id: str

    # 实验展示名称。
    name: str

    # 实验组列表,比如 control/treatment。
    groups: list[ExperimentGroup]

    # 是否启用。关闭时 assign 会返回默认 control。
    enabled: bool = True

    # 实验开始/结束时间预留字段。
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ExperimentGroup:
    """一个实验组。"""

    # 组名,例如 control、treatment_llm。
    name: str

    # 流量权重。多个组的 weight 会按比例切流。
    weight: int = 50

    # 该组对应的策略配置,例如 {"rerank": "llm"}。
    config: dict[str, Any] = field(default_factory=dict)

    # Thompson Sampling 的后验状态。
    # 初始 successes=1、failures=1 相当于 Beta(1,1),表示一开始没有偏见。
    successes: int = 1
    failures: int = 1


class ABTestEngine:
    """基于哈希分桶的 A/B 测试引擎,并支持 Thompson Sampling 动态分流。"""

    def __init__(self, bucket_count: int = 100):
        # 分桶数量。100 表示把用户分到 0-99 的桶。
        # engine = ABTestEngine(bucket_count=100)
        self.bucket_count = bucket_count

        # 保存所有实验定义,格式 {experiment_id: Experiment}。
        # self.experiments = {
        #     "recall_test": Experiment(...),
        #     "rerank_test": Experiment(...)
        # }
        self.experiments: dict[str, Experiment] = {}

        # 内存中的指标事件列表。生产环境可替换成 Kafka/数据库/埋点系统。
        self._metrics: list[dict[str, Any]] = []

        # 初始化项目默认实验。
        self._init_default_experiments()

    def _init_default_experiments(self):
        """注册默认实验。"""
        # 推荐策略实验: 对比规则重排和 LLM 重排。
        self.register_experiment(
            Experiment(
                id="rec_strategy",
                name="推荐策略实验",
                groups=[
                    ExperimentGroup(name="control", weight=50, config={"rerank": "rule_based"}),
                    ExperimentGroup(name="treatment_llm", weight=50, config={"rerank": "llm"}),
                ],
            )
        )

        # 文案风格实验: 对比正式风格和轻松风格。
        self.register_experiment(
            Experiment(
                id="copy_style",
                name="文案风格实验",
                groups=[
                    ExperimentGroup(name="formal", weight=50, config={"style": "formal"}),
                    ExperimentGroup(name="casual", weight=50, config={"style": "casual"}),
                ],
            )
        )

    def register_experiment(self, exp: Experiment):
        """注册或覆盖一个实验。"""
        self.experiments[exp.id] = exp

    def assign(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """使用一致性哈希把用户稳定分配到实验组。"""
        # 找不到实验或实验关闭时,默认返回 control。
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return {"group": "control", "config": {}}

        # 同一个 user_id + experiment_id 会得到同一个 bucket。
        bucket = self._hash_bucket(user_id, experiment_id)

        # 根据 bucket 和实验组权重映射到具体实验组。
        group = self._bucket_to_group(bucket, exp.groups)
        return {"group": group.name, "config": group.config}

    def assign_thompson(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
        """
        使用 Thompson Sampling 动态选择实验组。

        和固定 50/50 分流不同,它会让历史效果更好的组有更高概率被选中,
        但仍保留探索空间。
        """
        exp = self.experiments.get(experiment_id)
        if not exp or not exp.enabled:
            return {"group": "control", "config": {}}

        # 对每个实验组从 Beta(successes, failures) 中采样。
        samples = []
        for g in exp.groups:
            sample = np.random.beta(g.successes, g.failures)
            samples.append((sample, g))

        # 本次选择采样值最高的组。
        best = max(samples, key=lambda x: x[0])[1]
        return {"group": best.name, "config": best.config}

    def record_outcome(self, experiment_id: str, group_name: str, success: bool):
        """记录实验结果,更新 Thompson Sampling 的成功/失败次数。"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        # 找到对应实验组并更新后验计数。
        for g in exp.groups:
            if g.name == group_name:
                if success:
                    g.successes += 1
                else:
                    g.failures += 1
                break

    def record_metric(
        self,
        experiment_id: str,
        group_name: str,
        metric_name: str,
        value: float,
        user_id: str = "",
    ):
        """记录一个实验指标事件,例如 ctr/cvr/gmv。"""
        # 这里只存在内存里,用于 demo 和测试。
        self._metrics.append({
            "experiment_id": experiment_id,
            "group": group_name,
            "metric": metric_name,
            "value": value,
            "user_id": user_id,
            "timestamp": time.time(),
        })

    def get_stats(self, experiment_id: str) -> dict[str, Any]:
        """按实验组聚合指定实验的指标统计。"""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {}

        # 只取当前实验的指标。
        relevant = [m for m in self._metrics if m["experiment_id"] == experiment_id]

        # stats 中间结构:
        # {group: {metric_name: [value1, value2, ...]}}
        stats: dict[str, dict[str, list[float]]] = {}
        for m in relevant:
            grp = m["group"]
            metric = m["metric"]
            if grp not in stats:
                stats[grp] = {}
            if metric not in stats[grp]:
                stats[grp][metric] = []
            stats[grp][metric].append(m["value"])

        # 把指标列表聚合成 count/mean/std/min/max。把某一个实验之前记录的所有指标拿出来，
        # 先按“实验组 + 指标类型”分类，再计算每个指标的统计值，最后算出每个组的曝光、点击、加购、购买、CTR、CVR、GMV。
        result: dict[str, Any] = {}
        for grp, metrics in stats.items():
            result[grp] = {}
            for metric_name, values in metrics.items():
                arr = np.array(values)
                result[grp][metric_name] = {
                    "count": len(values),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }

            # 推荐实验常用漏斗指标。record_metric 会为曝光、点击、加购、购买分别写入 1。
            exposure_count = sum(metrics.get("exposure", []))
            click_count = sum(metrics.get("click", []))
            cart_count = sum(metrics.get("cart", []))
            purchase_count = sum(metrics.get("purchase", []))
            gmv = sum(metrics.get("gmv", []))
            result[grp]["funnel"] = {
                "exposure_count": exposure_count,
                "click_count": click_count,
                "cart_count": cart_count,
                "purchase_count": purchase_count,
                "ctr": click_count / exposure_count if exposure_count else 0.0,
                "cvr": purchase_count / exposure_count if exposure_count else 0.0,
                "gmv": gmv,
            }
        return result

    def _hash_bucket(self, user_id: str, experiment_id: str) -> int:
        """把用户稳定映射到 0 到 bucket_count-1 的桶。"""
        # 把用户 ID 和实验 ID 拼在一起,避免同一用户在所有实验里都落同一个桶。
        raw = f"{user_id}:{experiment_id}"

        # md5 用于稳定哈希。这里不是安全用途,只是分桶。
        h = hashlib.md5(raw.encode()).hexdigest()

        # 取前 8 位 hex 转整数再取模,得到桶号。
        return int(h[:8], 16) % self.bucket_count

    def _bucket_to_group(
        self, bucket: int, groups: list[ExperimentGroup]
    ) -> ExperimentGroup:
        """根据桶号和各实验组权重找到对应实验组。"""
        # 计算总权重,例如 50 + 50 = 100。
        total_weight = sum(g.weight for g in groups)

        # normalized_bucket 把 bucket 映射到权重区间。
        cumulative = 0
        normalized_bucket = bucket * total_weight / self.bucket_count

        # 按累计权重判断落在哪个组。
        for g in groups:
            cumulative += g.weight
            if normalized_bucket < cumulative:
                return g

        # 理论上不会走到这里,防止浮点数、配置异常之类的问题，保留兜底。
        return groups[-1]
