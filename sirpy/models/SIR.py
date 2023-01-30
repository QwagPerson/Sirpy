from typing import Any

from sirpy.models.abstractModel import AbstractModel, State, Transition


class SIR(AbstractModel):
    """
    Premade model SIR. This model is a simple SIR model.
    Has the same parameters as the AbstractModel.

    Parameters
    ----------
    name : str
        The name of the model. Useful when there are multiple instance models in the same
        scope.
    hyper_params : dict
        The hyper parameters of the model. They are not trained and are used to
        control the training process.
    train_params : dict
        The parameters of the model that are trained by the trainer. They are
        used to calculate the transition rates between states.
    static_params : dict
        The parameters of the model that are not trained but could be trained. They are useful
        when there is known constants in the dynamic of the model.
    args : Any
        Additional arguments used to ease inheritance.
    kwargs : Any
        Additional keyword arguments used to ease inheritance.

    Attributes
    ----------
    trained_data : ndarray
        The data that is used to train the model. It is a numpy array of shape (n, m) where n is the number of
        samples and m is the number of states used for training.
    test_data : ndarray
        The data that is used to test the model. It is a numpy array of shape (n, m) where n is the number of
        samples and m is the number of states used for testing.
    trained : bool
        A flag that indicates if the model is trained or not.
    shapes : list
        A list of the shapes of the train parameters. It is used to reshape the parameters after training.
    sizes : list
        A list of the sizes of the train parameters. It is used to reshape the parameters after training.


    """
    def __init__(self, name: str, hyper_params: dict, train_params: dict, static_params: dict, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(name, hyper_params, train_params, static_params, *args, **kwargs)

        # Let's create the states
        self.add_states([State("S"), State("I"), State("R")])

        # Now the transitions
        # Lambdas of the transition are functions of t,y,p -> ndarray
        self.add_transition(
            Transition(r"$\beta SI$", "S", "I",
                       lambda _, y, p: p.beta * y.S * y.I
                       )
        )

        self.add_transition(
            Transition(r"$\gamma I$", "I", "R",
                       lambda _, y, p: p.gamma * y.I
                       )
        )
