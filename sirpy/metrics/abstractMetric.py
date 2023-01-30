from abc import ABC
from typing import Any


class AbstractMetric(ABC):
    """
    The base of all metric. All metrics must inherit from this class.

    Parameters
    ----------
    name : str
        The name of the metric.
    """
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
