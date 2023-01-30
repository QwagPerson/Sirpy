from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from AbstractTrainer import AbstractTrainer
from abstractModel import AbstractModel
from sirpy.utils.attrDict import AttrDict


class LeastSquaresTrainer(AbstractTrainer):
    """
    Trains a model by minimizing the sum of squares of the residuals.

    Parameters
    ----------
    residual_fun : callable
        Function that computes the residuals. It must have the signature
        ``residual_fun(x: np.ndarray, trainer: LeastSquaresTrainer)``, where ``x`` is the current
        value of the independent variable and ``trainer`` is the trainer object.

    Attributes
    ----------
    results : least_squares
        The results of the optimization.
    param_attr_dict : AttrDict
        A AttrDict with the model's parameters. (AttrDict are accessed by dot notation)

    Notes
    -----
    Inherits all parameters and attributes from AbstractTrainer.
    """

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

    def update_param_attr_dict(self) -> None:
        """
        Updates the parameters of the model in the AttrDict.
        #TODO: Check if this fun is expensive and really necessary
        """
        self.param_attr_dict = AttrDict(
            **self.model.train_params,
            **self.model.static_params
        )

    def train(self, **kwargs: Any):
        """
        Trains the model by minimizing the sum of squares of the residuals.
        Parameters
        ----------
        kwargs : Any
            Keyword arguments to be passed to the ``least_squares`` function.

        Returns
        -------
        least_squares
            The results of the optimization.
        """
        if not self.model.trained:
            self.results = least_squares(
                self.residual_fun,
                self.model.get_train_params_as_list(),
                args=[self],
                **kwargs
            )
        self.model.trained = True
        return self.results

    # Asume que y está ordenado según el orden de states
    def compute_gradients(self, t : float, y: np.ndarray) -> np.ndarray:
        """
        Computes the gradients of the model's states.
        Parameters
        ----------
        t : float
            The current time.
        y : np.ndarray
            The current values of the states.

        Returns
        -------
        np.ndarray
            The gradients of the states.

        Notes
        -----
        This function assumes that the states in y array are ordered in the same way as the
        ``self.model.states`` list.
        """
        self.update_param_attr_dict()
        # if array is 1D, make it 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = AttrDict(**{i: y[j, :] for j, i in enumerate(self.model.states.keys())})
        lambda_vector = self.lambda_dict.values()
        # TODO: Check if this is the best way to do this? Maybe its too slow
        gradient = [f(t, y, self.param_attr_dict) for f in lambda_vector]
        # make array from gradient
        gradient = np.array(gradient)
        return gradient

    def solve_ode_system(self, **kwargs):
        """
        Uses the ``solve_ivp`` function to solve the ODE system.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the ``solve_ivp`` function.

        Returns
        -------
        np.ndarray # TODO: Check this
            The results of the ODE system.
        """
        res = solve_ivp(
            self.compute_gradients,
            self.model.get_param("time_space"),
            self.model.get_param("initial_condition"),
            vectorized=True,
            t_eval=self.model.get_param("time_range"),
            **kwargs
        )
        if not res.success:
            raise RuntimeError(f"ODE system could not be solved.\n"
                               f"Message: {res.message}\n"
                               f"Status: {res.status}\n"
                               f"Check if the model is well defined and try another starting point.\n"
                               "Good luck! :D")
        return solve_ivp(
            self.compute_gradients,
            self.model.get_param("time_space"),
            self.model.get_param("initial_condition"),
            vectorized=True,
            t_eval=self.model.get_param("time_range"),
            **kwargs
        )

    def calculate_curves(self):
        """
        Calculates the curve of all state the model.

        Returns
        -------
        np.ndarray
            The curves of the states. #TODO: Check this
        """
        return self.solve_ode_system().y.T

    def calculate_test_curves(self, **kwargs) -> np.ndarray:
        """
        Calculates the curve of all state the model for the test data.
        Parameters
        ----------
        kwargs: Any
            Keyword arguments to be passed to the ``solve_ivp`` function.

        Returns
        -------
        np.ndarray
            The curves of the states. #TODO: Check this
        """
        return solve_ivp(
            self.compute_gradients,
            self.model.get_param("test_time_space"),
            self.solve_ode_system().y.T[-1],
            vectorized=True,
            t_eval=self.model.get_param("test_time_range"),
            **kwargs
        ).y.T

def simple_residual_fun(x: np.ndarray, trainer: LeastSquaresTrainer):
    """
    Objetive function for least squares method. It is used to train the model.
    It is a simple residual function that calculates the difference between the
    model and the data.

    Parameters
    ----------
    x : np.ndarray
        The parameters of the model of that iteration of trained.

    trainer : LeastSquaresTrainer
        The trainer that is used to train the model. It can be used to access
        the data and to solve the ode system.

    Returns
    -------
    np.ndarray
        The difference between the model and the data. The target of the
        least squares method is to minimize this difference.
    Notes
    -----
    Im not sure if i should be using L2 norm or L1 norm here to calculate the difference.
    I think that L2 norm is better because it is more sensitive to outliers.
    """
    trainer.model.set_train_params_from_list(x)
    y_pred = trainer.calculate_curves()
    diff = trainer.model.train_data - y_pred
    return diff.flatten()
