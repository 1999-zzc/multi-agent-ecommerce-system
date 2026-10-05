"""A/B测试引擎单元测试"""

# sys/os 用来把 python/ 目录加入导入路径,方便直接运行本测试文件。
import sys
import os

# 将 tests/.. 也就是 python/ 目录加入模块搜索路径。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 导入被测试的 A/B 引擎和实验配置模型。
from services.ab_test import ABTestEngine, Experiment, ExperimentGroup


def test_consistent_assignment():
    """测试同一个用户会稳定分配到同一个实验组。"""
    engine = ABTestEngine()

    # 同一个 user_id 连续分配两次。
    group1 = engine.assign("user_001")
    group2 = engine.assign("user_001")

    # 期望两次分组一致,否则实验体验会不稳定。
    assert group1["group"] == group2["group"]


def test_distribution():
    """测试大量用户分桶后分布大致均衡。"""
    engine = ABTestEngine()

    # 统计 1000 个用户分别落到哪些组。
    counts: dict[str, int] = {}
    for i in range(1000):
        result = engine.assign(f"user_{i}")
        grp = result["group"]
        counts[grp] = counts.get(grp, 0) + 1

    # 默认是 50/50 分流,允许一定随机哈希偏差。
    for grp, count in counts.items():
        assert 300 < count < 700, f"Group {grp} has {count} users — too skewed"


def test_thompson_sampling():
    """测试 Thompson Sampling 的成功/失败计数会被正确更新。"""
    engine = ABTestEngine()

    # treatment_llm 连续记录成功。
    for _ in range(100):
        engine.record_outcome("rec_strategy", "treatment_llm", True)

    # control 连续记录失败。
    for _ in range(100):
        engine.record_outcome("rec_strategy", "control", False)

    # 从实验中取出两个组。
    exp = engine.experiments["rec_strategy"]
    treatment = next(g for g in exp.groups if g.name == "treatment_llm")
    control = next(g for g in exp.groups if g.name == "control")

    # treatment 的成功次数应该大于 control。
    assert treatment.successes > control.successes


def test_custom_experiment():
    """测试可以注册自定义实验。"""
    engine = ABTestEngine()

    # 注册一个 Prompt 模板实验,两个组权重 30/70。
    engine.register_experiment(
        Experiment(
            id="prompt_test",
            name="Prompt模板实验",
            groups=[
                ExperimentGroup(name="template_a", weight=30),
                ExperimentGroup(name="template_b", weight=70),
            ],
        )
    )

    # 分配用户到自定义实验。
    result = engine.assign("user_999", "prompt_test")

    # 结果必须属于注册过的两个实验组之一。
    assert result["group"] in ("template_a", "template_b")


def test_metrics_recording():
    """测试实验指标记录和聚合统计。"""
    engine = ABTestEngine()

    # 记录三条 CTR 指标,两条 control,一条 treatment_llm。
    engine.record_metric("rec_strategy", "control", "ctr", 0.05, "user_001")
    engine.record_metric("rec_strategy", "control", "ctr", 0.08, "user_002")
    engine.record_metric("rec_strategy", "treatment_llm", "ctr", 0.12, "user_003")

    # 获取聚合统计。
    stats = engine.get_stats("rec_strategy")

    # control 组应该存在,且 ctr 指标数量为 2。
    assert "control" in stats
    assert stats["control"]["ctr"]["count"] == 2


if __name__ == "__main__":
    # 允许不用 pytest,直接 `python tests/test_ab_test.py` 运行这些测试。
    test_consistent_assignment()
    test_distribution()
    test_thompson_sampling()
    test_custom_experiment()
    test_metrics_recording()
    print("All A/B test engine tests passed!")
