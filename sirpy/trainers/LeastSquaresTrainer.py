from typing import Any, Callable

import numpy as np

from AbstractTrainer import AbstractTrainer
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from abstractModel import AbstractModel
import pandas as pd

# define a lambda function that takes y, t and return 0 and assing it to a variable named null_lambda
null_lambda = lambda y, t: 0

# make a function that takes two lambda functions and add em togheher
def add_functions(f1, f2):
    return lambda y, t: f1(y, t) + f2(y, t)


# make a function that takes a list of lambda functions and difference em togheher
def difference_functions(f1, f2):
    return lambda y, t: f1(y, t) - f2(y, t)


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
        self.lambda_dict = None

    def train(self, **kwargs: Any):
        self.results = least_squares(
            self.residual_fun,
            self.model.get_train_params_as_list(),
            args=[self],
            **kwargs
        )
        self.model.trained = True
        return self.results

    def compute_lamda_dict(self):
        self.lambda_dict = {i: null_lambda for i in self.model.states.keys()}
        for i, transition in enumerate(self.model.transitions):
            self.lambda_dict[transition.left] = difference_functions(
                self.lambda_dict[transition.left],
                transition.fun
            )
            self.lambda_dict[transition.right] = add_functions(
                self.lambda_dict[transition.right],
                transition.fun
            )

    # Asume que y está ordenado según el orden de states
    def compute_gradients(self, t, y):
        if self.lambda_dict is None:
            self.compute_lamda_dict()
        # if array is 1D, make it 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = {i: y[j, :] for j, i in enumerate(self.model.states.keys())}
        lambda_vector = self.lambda_dict.values()
        gradient = [f(y, t) for f in lambda_vector]
        # make array from gradient
        gradient = np.array(gradient)
        return gradient

    def solve_ode_system(self, **kwargs):
        return solve_ivp(
            self.compute_gradients,
            self.model.get_param("time_space"),
            self.model.get_param("initial_condition"),
            vectorized=True,
            t_eval = self.model.get_param("time_range"),
            **kwargs
        )

    def calculate_curves(self):
        return self.solve_ode_system().y.T

    def plot_curves(self):
        return pd.DataFrame(self.calculate_curves(), columns=list(self.model.states.keys())).plot()
