from typing import Any
from pymc.distributions import Distribution
from sirpy.parameter.AbstractParameter import AbstractParameter
from BayesianTrainer import BayesianTrainer


class BayesianParameter(AbstractParameter):
    def __init__(self, name: str, value, *args: Any, **kwargs: Any):
        super().__init__(name, value, *args, **kwargs)

    def add_to_bayesian_model(self, bayes_trainer: BayesianTrainer):
        bayes_trainer.add_distribution_param(self.name, self.value, **self.kwargs)