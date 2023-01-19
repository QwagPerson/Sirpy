from typing import Any, Callable
from AbstractTrainer import AbstractTrainer
from scipy.optimize import least_squares
from scipy.integrate import odeint
from abstractModel import AbstractModel
import pandas as pd


class LeastSquaresTrainer(AbstractTrainer):
    def __init__(self,
                 model: AbstractModel,
                 residual_fun: Callable = None,
                 *args: Any,
                 **kwargs: Any
                 ) -> None:
        super().__init__(model, *args, **kwargs)
        self.residual_fun = self.simple_residual_fun if residual_fun is None else residual_fun

    def train(self, **kwargs: Any):
        self.results = least_squares(
            self.residual_fun,
            list(self.model.train_params.values()),  # Change this, a lot.
            **kwargs
        )
        return self.results

    def simple_residual_fun(self, x):
        self.model.save_new_target_params(x)
        y_pred = self.calculate_curves()
        diff = self.model.train_data - y_pred
        return diff.flatten()

    # Asume que y está ordenado según el orden de states
    def compute_gradients(self, y: list, t: float) -> list:
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

    def solve_ode_system(self):
        return odeint(
            self.compute_gradients,
            self.model.get_param("initial_condition"),
            self.model.get_param("time_space")
        )

    def calculate_curves(self):
        return self.solve_ode_system()

    def plot_curves(self):
        return pd.DataFrame(self.solve_ode_system(), columns=list(self.model.states.keys())).plot()
