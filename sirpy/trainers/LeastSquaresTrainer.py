from typing import Any, Callable

import numpy as np

from AbstractTrainer import AbstractTrainer
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from abstractModel import AbstractModel
import pandas as pd
from sirpy.utils.attrDict import AttrDict


def simple_residual_fun(x, trainer):
    trainer.model.set_train_params_from_list(x)
    y_pred = trainer.calculate_curves()
    diff = trainer.model.train_data - y_pred
    return diff.flatten()


class LeastSquaresTrainer(AbstractTrainer):
    def __init__(self,
                 model: AbstractModel,
                 residual_fun: Callable = None,
                 *args: Any,
                 **kwargs: Any
                 ) -> None:
        super().__init__(model, *args, **kwargs)
        self.residual_fun = simple_residual_fun if residual_fun is None else residual_fun
        self.param_attr_dict = AttrDict(
            **self.model.train_params,
            **self.model.static_params
        )

    def update_param_attr_dict(self):
        self.param_attr_dict = AttrDict(
            **self.model.train_params,
            **self.model.static_params
        )

    def train(self, **kwargs: Any):
        self.results = least_squares(
            self.residual_fun,
            self.model.get_train_params_as_list(),
            args=[self],
            **kwargs
        )
        self.model.trained = True
        return self.results

    # Asume que y está ordenado según el orden de states
    def compute_gradients(self, t, y):
        self.update_param_attr_dict()
        # if array is 1D, make it 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = AttrDict(**{i: y[j, :] for j, i in enumerate(self.model.states.keys())})
        lambda_vector = self.lambda_dict.values()
        gradient = [f(t, y, self.param_attr_dict) for f in lambda_vector]
        # make array from gradient
        gradient = np.array(gradient)
        return gradient

    def solve_ode_system(self, **kwargs):
        return solve_ivp(
            self.compute_gradients,
            self.model.get_param("time_space"),
            self.model.get_param("initial_condition"),
            vectorized=True,
            t_eval=self.model.get_param("time_range"),
            **kwargs
        )

    def calculate_curves(self):
        return self.solve_ode_system().y.T

