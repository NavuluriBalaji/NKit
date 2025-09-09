from typing import Any, Dict, List, Optional


class Memory:
    """A tiny in-memory key-value store for agent state.

    Designed to be simple and synchronous. Suitable for unit tests and
    as a drop-in lightweight memory for agents.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def append(self, key: str, value: Any) -> None:
        if key not in self._store or not isinstance(self._store[key], list):
            self._store[key] = []
        self._store[key].append(value)

    def clear(self) -> None:
        self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._store)


__all__ = ["Memory"]
