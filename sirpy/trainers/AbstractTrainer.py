from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from abstractModel import AbstractModel


class AbstractTrainer(ABC):
    def __init__(self, model: AbstractModel, *args: Any, **kwargs: Any) -> None:
        self.model = model
        self.results = None
        pass

    @abstractmethod
    def train(self, **kwargs: Any):
        pass

    @abstractmethod
    def solve_ode_system(self):
        pass

    def calculate_curves(self):
        return self.solve_ode_system()

    def plot_curves(self):
        return pd.DataFrame(self.solve_ode_system(), columns=list(self.model.states.keys())).plot()