from abc import ABC, abstractmethod
from typing import Any

from abstractModel import AbstractModel


class AbstractCalibrator(ABC):
    def __init__(self, model: AbstractModel, *args: Any, **kwargs: Any) -> None:
        self.model = model
        self.results = None
        pass

    @abstractmethod
    def calibrate(self, **kwargs: Any):
        pass