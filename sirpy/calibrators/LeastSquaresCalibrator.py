from typing import Any, Callable
from AbstractCalibrator import AbstractCalibrator
from scipy.optimize import least_squares

from abstractModel import AbstractModel


class LeastSquaresCalibrator(AbstractCalibrator):
    def __init__(self,
                 model: AbstractModel,
                 residual_fun: Callable = None,
                 *args: Any,
                 **kwargs: Any
                 ) -> None:
        super().__init__(model, *args, **kwargs)
        self.residual_fun = self.simple_residual_fun if residual_fun is None else residual_fun

    def calibrate(self, **kwargs: Any):
        self.results = least_squares(
            self.residual_fun,
            list(self.model.target_params.values()), # Change this, a lot.
            **kwargs
        )
        return self.results

    def simple_residual_fun(self, x):
        self.model.save_new_target_params(x)
        y_pred = self.model.calculate_curves()
        diff = self.model.train_data - y_pred
        return diff.flatten()