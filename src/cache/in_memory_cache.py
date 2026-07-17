import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


# Represent a cached RAG gateway response and its expiry timestamp.
@dataclass(slots=True)
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class InMemoryTTLCache(Generic[T]):
    """
    Small process-local TTL cache.

    This is intentionally simple: it is fast, dependency-free, and easy to
    replace with Redis later because callers only need get/set semantics.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 900) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than zero.")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")

        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, CacheEntry[T]] = OrderedDict()

    # Return cached values only while they remain inside the TTL window.
    def get(self, key: str) -> T | None:
        entry = self._items.get(key)
        if entry is None:
            return None

        if entry.expires_at <= time.time():
            self._items.pop(key, None)
            return None

        self._items.move_to_end(key)
        return entry.value

    # Store a bounded cache entry and evict oldest values when needed.
    def set(self, key: str, value: T) -> None:
        self._items[key] = CacheEntry(
            value=value,
            expires_at=time.time() + self._ttl_seconds,
        )
        self._items.move_to_end(key)

        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

    # Clear all process-local cached RAG responses.
    def clear(self) -> None:
        self._items.clear()
