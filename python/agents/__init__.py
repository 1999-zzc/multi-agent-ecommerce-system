# 这个文件是 agents 包的统一出口。
# 有了它,其他模块就可以写 `from agents import UserProfileAgent`,
# 不需要知道每个 Agent 分别放在哪个具体文件里。
from .user_profile_agent import UserProfileAgent
from .product_rec_agent import ProductRecAgent
from .subsidy_decision_agent import SubsidyDecisionAgent
from .inventory_agent import InventoryAgent
from .base_agent import BaseAgent

# __all__ 控制 `from agents import *` 时允许导出的名字。
# 这里把项目中对外可用的 Agent 类都列出来,形成清晰的包边界。
__all__ = [
    "BaseAgent",
    "UserProfileAgent",
    "ProductRecAgent",
    "SubsidyDecisionAgent",
    "InventoryAgent",
]
