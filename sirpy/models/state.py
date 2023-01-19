from typing import Any

import abstractModel


class State:
    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.val = None
        self.model = None
        self.grad = None

    def set_model(self, aModel: abstractModel) -> None:
        self.model = aModel

    def reset_gradient(self):
        self.grad = 0.0

    def set_value(self, val: float) -> None:
        self.val = val

