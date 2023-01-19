from abc import ABC, abstractmethod
from typing import Any


class AbstractMetric(ABC):
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
