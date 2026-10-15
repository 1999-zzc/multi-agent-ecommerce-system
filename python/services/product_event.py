from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ProductEvent:
    """
    商品变化事件。

    真实生产中事件一般来自 Kafka/RabbitMQ。
    当前项目不引入消息队列,只保留事件对象和处理逻辑。
    """
    event_type: str
    product_id: str
