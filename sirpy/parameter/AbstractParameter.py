from abc import ABC, abstractmethod
from typing import Any


class AbstractParameter(ABC):
    def __init__(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.value = value
        self.model = None
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def add_to_bayesian_model(self, bayes_trainer):
        pass
