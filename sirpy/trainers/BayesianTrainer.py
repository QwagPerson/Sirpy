from typing import Any, Callable

from pymc import Distribution
from pymc.ode import DifferentialEquation

from AbstractTrainer import AbstractTrainer
from scipy.optimize import least_squares
from scipy.integrate import odeint
from abstractModel import AbstractModel
import pandas as pd
import pymc as pm


class BayesianTrainer(AbstractTrainer):
    def __init__(self,
                 model_structure: AbstractModel,
                 *args: Any,
                 **kwargs: Any
                 ) -> None:
        super().__init__(model_structure, *args, **kwargs)
        self.ode_result = None
        self.bayesian_model = pm.Model()
        # Transforming floats into parameters objects
        self.model.make_parameters_object()
        self.build_bayesian_model_representation()

    def build_bayesian_model_representation(self):
        # add params as distribution of the model
        for param in self.model.train_params.values():
            param.add_to_bayesian_model(self)

        # add deterministic params
        for param in self.model.static_params.values():
            param.add_to_bayesian_model(self)

    def add_distribution_param(self, name: str, a_distribution, *args: Any, **kwargs: Any):
        with self.bayesian_model:
            self.model.train_params[name] = a_distribution(name, *args, **kwargs)

    # Check if this is really necessary
    def add_deterministic_param(self, name: str, value: float):
        with self.bayesian_model:
            self.model.static_params[name] = pm.Deterministic(name, value)

    def compute_gradients(self, y, t, n_theta) -> list:
        ...

    def train(self, **kwargs: Any):
        ...

    def solve_ode_system(self):
        ...
