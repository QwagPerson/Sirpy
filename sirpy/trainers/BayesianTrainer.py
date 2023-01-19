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
        # FIXME: Adapt senode to ode model
        self.ode_model = DifferentialEquation(
            func=self.compute_gradients,
            times=self.model.hyper_params["time_slice"],
            n_states=len(self.model.states.keys()),
            n_theta=len(self.model.train_params.keys()),  # FIXME: Check if it's correct
            t0=0  # FIXME: add t0 to hyper params
        )


    def build_bayesian_model_representation(self):
        # add params as distribution of the model
        for param in self.model.train_params.values():
            param.add_to_bayesian_model(self)

        # add deterministic params
        for param in self.model.static_params.values():
            param.add_to_bayesian_model(self)

    def compute_gradients(self, y, t, n_theta) -> list:
        gradients = []
        if len(y) != len(self.model.states):
            raise ValueError(f"Length mismatch between y and states. ({len(y)}-{len(self.model.states)})")

        for val, a_state in zip(y, self.model.states.values()):
            a_state.reset_gradient()
            a_state.set_value(val)

        for a_transition in self.model.transitions:
            a_transition.compute_gradients(t)

        for a_state in self.model.states.values():
            gradients.append(a_state.grad)

        return gradients

    def train(self, **kwargs: Any):
        with self.bayesian_model:
            self.ode_result = self.ode_model(y0=self.model.initial_conditions
                                             ,theta=self.model.train_params.values())

        return self.results

    def solve_ode_system(self):
        pass

    def add_distribution_param(self, name: str, a_distribution, *args: Any, **kwargs: Any):
        with self.bayesian_model:
            a_distribution(name, *args, **kwargs)

    def add_deterministic_param(self, name: str, value: float):
        with self.bayesian_model:
            pm.Deterministic(name, value)
