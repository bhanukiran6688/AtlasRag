import pytest
import time
from src.cache.in_memory_cache import InMemoryTTLCache, CacheEntry


class TestCacheEntry:
    def test_cache_entry_stores_value_and_expiry(self):
        entry = CacheEntry(value="test_value", expires_at=time.time() + 100)
        assert entry.value == "test_value"
        assert entry.expires_at > time.time()


class TestInMemoryTTLCache:
    def test_cache_raises_error_for_zero_max_size(self):
        with pytest.raises(ValueError, match="max_size must be greater than zero"):
            InMemoryTTLCache(max_size=0, ttl_seconds=100)

    def test_cache_raises_error_for_negative_max_size(self):
        with pytest.raises(ValueError, match="max_size must be greater than zero"):
            InMemoryTTLCache(max_size=-1, ttl_seconds=100)

    def test_cache_raises_error_for_zero_ttl(self):
        with pytest.raises(ValueError, match="ttl_seconds must be greater than zero"):
            InMemoryTTLCache(max_size=100, ttl_seconds=0)

    def test_cache_raises_error_for_negative_ttl(self):
        with pytest.raises(ValueError, match="ttl_seconds must be greater than zero"):
            InMemoryTTLCache(max_size=100, ttl_seconds=-1)

    def test_get_returns_none_for_nonexistent_key(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        result = cache.get("nonexistent_key")
        assert result is None

    def test_set_and_get_stores_and_retrieves_value(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_returns_none_for_expired_entry(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        time.sleep(1.1)
        result = cache.get("key1")
        assert result is None

    def test_get_removes_expired_entry(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=1)
        cache.set("key1", "value1")
        time.sleep(1.1)
        cache.get("key1")
        assert "key1" not in cache._items

    def test_set_updates_existing_key(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        result = cache.get("key1")
        assert result == "value2"

    def test_set_moves_key_to_end(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key1", "value1_updated")

        # key1 should be at the end (most recently used)
        last_key = list(cache._items.keys())[-1]
        assert last_key == "key1"

    def test_get_moves_key_to_end(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")

        # key1 should be at the end (most recently used)
        last_key = list(cache._items.keys())[-1]
        assert last_key == "key1"

    def test_cache_evicts_oldest_when_full(self):
        cache = InMemoryTTLCache(max_size=3, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_evicts_multiple_when_over_limit(self):
        cache = InMemoryTTLCache(max_size=2, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_clear_removes_all_entries(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache._items) == 0

    def test_cache_handles_different_value_types(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=100)

        cache.set("string", "value")
        cache.set("int", 123)
        cache.set("float", 45.67)
        cache.set("bool", True)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"key": "value"})

        assert cache.get("string") == "value"
        assert cache.get("int") == 123
        assert cache.get("float") == 45.67
        assert cache.get("bool") is True
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"key": "value"}

    def test_cache_with_custom_max_size(self):
        cache = InMemoryTTLCache(max_size=5, ttl_seconds=100)
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Should only keep the last 5 entries
        assert len(cache._items) == 5
        assert cache.get("key5") == "value5"
        assert cache.get("key9") == "value9"
        assert cache.get("key0") is None

    def test_cache_with_custom_ttl(self):
        cache = InMemoryTTLCache(max_size=10, ttl_seconds=2)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(2.1)
        assert cache.get("key1") is None
