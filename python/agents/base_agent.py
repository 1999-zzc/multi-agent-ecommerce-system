from __future__ import annotations
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from models.schemas import AgentResult

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
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
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
        start = time.perf_counter()
        self._call_count += 1
        try:
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except asyncio.TimeoutError:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            timeout_error = TimeoutError(f"agent timed out after {self.timeout} seconds")
            logger.error("agent.timeout", agent=self.name, timeout_seconds=self.timeout)
            return self._fallback(latency_ms, timeout_error, **kwargs)
        except Exception as exc:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("agent.failed", agent=self.name, error=str(exc))
            return self._fallback(latency_ms, exc, **kwargs)

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        return await self._retry_operation(lambda: self._execute(**kwargs))

    async def _retry_operation(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """为分阶段任务复用超时和指数退避重试逻辑。"""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _inner():
            return await asyncio.wait_for(
                operation(),
                timeout=self.timeout,
            )
        return await _inner()

    def _fallback(self, latency_ms: float, exc: Exception, **kwargs: Any) -> AgentResult:
        """
        失败兜底结果。

        这里返回的是通用兜底,只说明哪个 Agent 失败了。
        如果某个子类需要更业务化的兜底,可以重写这个方法。
        kwargs 会保留原始入参，便于子类兜底时使用 user_id 等业务字段。
        """
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
