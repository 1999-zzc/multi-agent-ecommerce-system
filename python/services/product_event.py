from __future__ import annotations

# dataclass 用来定义轻量事件对象。
from dataclasses import dataclass


@dataclass
class ProductEvent:
    """
    商品变化事件。

    真实生产中事件一般来自 Kafka/RabbitMQ。
    当前项目不引入消息队列,只保留事件对象和处理逻辑。
    """

    # 事件类型: product_created/product_updated/product_deleted/product_offline。
    event_type: str

    # 发生变化的商品 ID。
    product_id: str
