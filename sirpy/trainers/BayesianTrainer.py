from typing import Any

import pymc as pm
import sunode.wrappers.as_pytensor

from AbstractTrainer import AbstractTrainer
from abstractModel import AbstractModel


# Since this is a WIP i have not documented everything
# The idea is to have a trainer that uses bayesian inference to train the model
# The trainer will use pymc to do the inference and sunode to solve the ode system
# The trainer will have a bayesian model representation of the model

class DistributionParameter:
    def __init__(self, name: str, distribution: pm.distributions, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.distribution = distribution
        self.args = args
        self.kwargs = kwargs


class BayesianTrainer(AbstractTrainer):
    def __init__(self,
                 model_structure: AbstractModel,
                 *args: Any,
                 **kwargs: Any
                 ) -> None:
        super().__init__(model_structure, *args, **kwargs)
        self.bayesian_model = pm.Model()
        self.build_bayesian_model_representation()
        self.rhs = lambda t, y, p: {k: v(t, y, p) for k, v in zip(self.lambda_dict.keys(), self.lambda_dict.values())}
        self.y0 = None

    def build_bayesian_model_representation(self):
        with self.bayesian_model:
            # define y0 as a empty dict
            self.y0 = {}

            # add initial conditions as distribution of the model
            for c in self.model.p("initial_conditions"):
                self.y0[c.name] = (c.distribution(c.name, *c.args, **c.kwargs), ())

            if self.y0.keys() != self.model.states.keys():
                raise Exception("y0 and self.states have not the same keys")

            # make a copy of the model.train_params.values and delete the
            # one whose name is "initial_conditions"
            train_params = self.model.train_params.values()
            train_params = [i for i in train_params if i.name != "initial_conditions"]

            # add params as distribution of the model
            for p in train_params:
                self.model.train_params[p.name] = (p.distribution(p.name, *p.args, **p.kwargs), ())

    def solve_ode_system(self):
        with self.bayesian_model:
            y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
                y0=self.y0,
                params={**self.model.static_params, **self.model.train_params},
                # A functions that computes the right-hand-side of the ode using
                # sympy variables.
                rhs=self.rhs,
                # The time points where we want to access the solution
                tvals=self.model.p("time_range"),
                t0=self.model.p("time_range")[0],
            )

    def train(self, **kwargs: Any):
        ...

    def calculate_curves(self):
        ...
