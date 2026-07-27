"""In-memory LRU cache of loaded model artifacts — the fast what-if path."""

import threading
from collections import OrderedDict
from typing import Any, Callable

DEFAULT_CAPACITY = 8


class ModelCache:
    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self._capacity = capacity
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        loaded = loader()
        with self._lock:
            self._items[key] = loaded
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return loaded
