import threading
from typing import Any


class SingletonMeta(type):
    """Metaclass for creating singleton classes. Used to keep global configs shared between instances."""

    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in SingletonMeta._instances:
            with SingletonMeta._lock:
                if cls not in SingletonMeta._instances:
                    instance = super().__call__(*args, **kwargs)
                    SingletonMeta._instances[cls] = instance
        return SingletonMeta._instances[cls]
