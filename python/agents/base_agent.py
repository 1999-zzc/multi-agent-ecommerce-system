from __future__ import annotations             #让类型注解延迟解析

# time 用来统计每个 Agent 从开始执行到返回结果的耗时。
import time

# asyncio.wait_for 用于在 Agent 调用超时后取消协程，防止慢 LLM 拖住整条推荐链路。
import asyncio

# ABC/abstractmethod 用来定义“抽象基类”。
# BaseAgent 只规定公共流程,具体业务逻辑必须由子类实现。
from abc import ABC, abstractmethod

# Any 表示任意类型。不同 Agent 需要的参数不同,所以这里用 Any 接收。
from typing import Any, Awaitable, Callable

# structlog 是结构化日志库,适合记录 agent=xxx、latency_ms=xxx 这种可检索字段。
import structlog

# tenacity 是重试库:
# - retry: 给函数加自动重试
# - stop_after_attempt: 设置最多尝试次数
# - wait_exponential: 每次失败后按指数退避等待
from tenacity import retry, stop_after_attempt, wait_exponential

# AgentResult 是所有 Agent 统一返回的数据结构。
# 统一结果结构后,Supervisor 就能用同一种方式处理不同 Agent 的结果。
from models.schemas import AgentResult

# 全局 logger,本文件所有日志都通过它输出。
logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    所有具体 Agent 的父类。

    这个类不做具体业务,只负责 Agent 的通用运行生命周期:
    1. 记录调用次数
    2. 统计耗时
    3. 失败自动重试
    4. 重试后仍失败时返回 fallback 结果
    5. 计算错误率
    """

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        # Agent 名称,用于日志、监控和返回结果标识。
        self.name = name

        # 单次 Agent 执行允许的最大时长，单位秒。
        # _retry_execute() 会用 asyncio.wait_for 真正强制这个上限。
        self.timeout = timeout

        # 最大尝试次数。tenacity 中这个次数包含第一次执行。
        self.max_retries = max_retries

        # 记录 run() 被调用的总次数,后面用于计算 error_rate。
        self._call_count = 0

        # 记录最终失败的次数。只有重试也失败并进入 fallback 时才会加 1。
        self._error_count = 0

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """
        子类必须实现的核心业务逻辑。

        例如:
        - UserProfileAgent 在这里生成用户画像
        - ProductRecAgent 在这里召回和重排商品
        - SubsidyDecisionAgent 在这里检索政策并选择补贴商品
        - InventoryAgent 在这里检查候选商品库存
        """

    async def run(self, **kwargs: Any) -> AgentResult:
        """
        Agent 的公开入口。

        外部代码应该调用 run(),不要直接调用 _execute()。
        因为 run() 会统一补上计时、重试、日志和失败兜底。
        """
        # 记录开始时间,后面用当前时间减去 start 计算耗时。
        start = time.perf_counter()

        # 每调用一次 run(),总调用次数加 1。
        self._call_count += 1

        try:
            # 调用带重试能力的内部执行方法。
            # 如果 _execute 临时失败,tenacity 会自动再次尝试。
            result = await self._retry_execute(**kwargs)

            # 成功后把耗时写回结果对象,单位从秒转换成毫秒。
            result.latency_ms = (time.perf_counter() - start) * 1000

            # 记录成功日志,方便后续观察每个 Agent 的耗时。
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except asyncio.TimeoutError:
            # 超时也视为最终失败。这里单独处理，日志和返回错误会更容易排查。
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            timeout_error = TimeoutError(f"agent timed out after {self.timeout} seconds")
            logger.error("agent.timeout", agent=self.name, timeout_seconds=self.timeout)
            return self._fallback(latency_ms, timeout_error, **kwargs)
        except Exception as exc:
            # 只有“所有重试都失败”时才会进入这里。
            self._error_count += 1

            # 即使失败也记录耗时,这样监控里能看到失败请求消耗了多久。
            latency_ms = (time.perf_counter() - start) * 1000

            # 记录失败日志。str(exc) 会把异常转换成可读的错误文本。
            logger.error("agent.failed", agent=self.name, error=str(exc))

            # 返回一个合法但标记为失败的降级结果,避免整个流程直接崩掉。
            return self._fallback(latency_ms, exc, **kwargs)

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        # 将子类核心逻辑交给通用的“超时 + 重试”包装器执行。
        return await self._retry_operation(lambda: self._execute(**kwargs))

    async def _retry_operation(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """为分阶段任务复用超时和指数退避重试逻辑。"""
        # tenacity 的 retry 装饰器只能包函数,所以这里定义一个内部函数 _inner。
        @retry(
            # 最多尝试 self.max_retries 次。
            stop=stop_after_attempt(self.max_retries),

            # 指数退避等待: 失败后至少等 0.5 秒,最多等 4 秒。
            # 这样可以减少外部服务短暂抖动时的连续冲击。
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),

            # 重试次数耗尽后继续抛出最后一次异常,让 run() 的 except 处理 fallback。
            reraise=True,
        )
        async def _inner():
            # 超时时 wait_for 会取消当前协程并抛出 asyncio.TimeoutError，随后由 retry 重试。
            return await asyncio.wait_for(
                operation(),
                timeout=self.timeout,
            )

        # 调用被 retry 包装后的内部函数。
        return await _inner()

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        """
        失败兜底结果。

        这里返回的是通用兜底,只说明哪个 Agent 失败了。
        如果某个子类需要更业务化的兜底,可以重写这个方法。
        kwargs 会保留原始入参，便于子类兜底时使用 user_id 等业务字段。
        """
        return AgentResult(
            # 告诉调用方是哪个 Agent 返回的结果。
            agent_name=self.name,

            # success=False 表示这不是正常业务结果。
            success=False,

            # 保留失败前消耗的时间。
            latency_ms=latency_ms,

            # 把异常信息放到结果里,方便接口调用方或日志排查。
            error=str(exc),

            # 失败结果没有可信业务内容,所以置信度为 0。
            confidence=0.0,
        )

    @property
    def error_rate(self) -> float:
        # 没有调用过时不能做除法,否则会除以 0。
        if self._call_count == 0:
            return 0.0

        # 错误率 = 最终失败次数 / 总调用次数。
        return self._error_count / self._call_count
