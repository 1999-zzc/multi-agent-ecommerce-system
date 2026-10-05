# config 包的统一出口。
# 其他文件可以直接写 `from config import get_settings`,
# 不需要关心配置类具体定义在 settings.py 里。
from .settings import Settings, get_settings

# 控制 `from config import *` 时导出的对象。
__all__ = ["Settings", "get_settings"]
