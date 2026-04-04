"""Tests for SingletonMeta metaclass."""

from uipath.llm_client.settings.utils import SingletonMeta


class TestSingletonMeta:
    """Tests for SingletonMeta metaclass."""

    def test_singleton_creates_single_instance(self):
        """Test singleton creates only one instance when no cache key is defined."""

        class TestSingleton(metaclass=SingletonMeta):
            def __init__(self, value: int):
                self.value = value

        instance1 = TestSingleton(1)
        instance2 = TestSingleton(2)

        assert instance1 is instance2
        assert instance1.value == 1  # First value is retained

    def test_different_classes_have_different_instances(self):
        """Test different singleton classes have separate instances."""

        class SingletonA(metaclass=SingletonMeta):
            pass

        class SingletonB(metaclass=SingletonMeta):
            pass

        a = SingletonA()
        b = SingletonB()

        assert a is not b

    def test_keyed_singleton_same_key_reuses_instance(self):
        """Test that same cache key returns the same instance."""

        class KeyedSingleton(metaclass=SingletonMeta):
            def __init__(self, key: str, value: int):
                self.key = key
                self.value = value

            @classmethod
            def _singleton_cache_key(cls, key: str, value: int) -> tuple:
                return (key,)

        a = KeyedSingleton("k1", 10)
        b = KeyedSingleton("k1", 20)
        assert a is b
        assert a.value == 10  # First value retained

    def test_keyed_singleton_different_key_creates_new_instance(self):
        """Test that different cache keys create separate instances."""

        class KeyedSingleton2(metaclass=SingletonMeta):
            def __init__(self, key: str, value: int):
                self.key = key
                self.value = value

            @classmethod
            def _singleton_cache_key(cls, key: str, value: int) -> tuple:
                return (key,)

        a = KeyedSingleton2("k1", 10)
        b = KeyedSingleton2("k2", 20)
        assert a is not b
        assert a.value == 10
        assert b.value == 20
