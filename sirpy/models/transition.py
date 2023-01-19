from typing import Any, Callable

import abstractModel


# Son simetricas
class SimpleTransition:
    def __init__(self, name: str, left: str, right: str, fun: Callable, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.left = left
        self.right = right
        self.fun = fun
        self.model = None

    def compute_gradients(self, t):
        amount = self.fun(t)
        self.model.states[self.left].grad -= amount
        self.model.states[self.right].grad += amount

    def set_model(self, aModel: abstractModel) -> None:
        self.model = aModel
