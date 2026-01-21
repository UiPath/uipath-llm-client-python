from typing import Any


class SingletonMeta[T](type):
    """Metaclass for creating singleton classes. Used to keep global configs shared between instances."""

    _instances: dict[Any, Any] = {}

    def __call__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        instances = SingletonMeta._instances
        if cls not in instances:
            instance = super().__call__(*args, **kwargs)
            instances[cls] = instance
        return instances[cls]
