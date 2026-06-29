import threading
from typing import Any


class SingletonMeta(type):
    """Metaclass for creating singleton classes keyed by (class, cache_key).

    Classes using this metaclass can define a ``_singleton_cache_key`` classmethod
    that derives a hashable key from the constructor arguments.  When the same
    key is seen again the cached instance is returned and ``__init__`` is
    **not** re-invoked.

    If the class does not define ``_singleton_cache_key``, the class itself is
    used as the sole key (original singleton-per-class behaviour).

    Used to share access-tokens / auth state between multiple HTTP clients that
    are configured with the same credentials.
    """

    _instances: dict[tuple[type, Any], Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        key_fn = getattr(cls, "_singleton_cache_key", None)
        if key_fn is not None:
            cache_key = (cls, key_fn(*args, **kwargs))
        else:
            cache_key = (cls, None)

        if cache_key not in SingletonMeta._instances:
            with SingletonMeta._lock:
                if cache_key not in SingletonMeta._instances:
                    instance = super().__call__(*args, **kwargs)
                    SingletonMeta._instances[cache_key] = instance
        return SingletonMeta._instances[cache_key]
