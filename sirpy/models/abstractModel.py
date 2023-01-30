from abc import ABC
from typing import Any, Callable, Union

import networkx as nx
import numpy as np
from matplotlib.axes import Axes


class State:
    """Represents one node in the graph of a model.

    Parameters
    ----------
    name : str
        Tag of the node used for printing and plotting.
    """
    def __init__(self, name: str) -> None:

        self.name = name


class Transition:
    """Represents one edge in the graph of a model. It stores functions that are used to calculate the transition rate.

    Parameters
    ----------
    name : str
        Tag of the edge used for printing and plotting.
    left : str
        The tag of node from which the edge starts. It is used to identify the node from which the flow starts.
    right : str
        The tag of node to which the edge ends. It is used to identify the node to which the flow ends.
    fun : Callable
        The function that is used to calculate the transition rate. It must take as arguments the time, the state of the
        model and the parameters of the model.
    symmetrical : bool, optional
        If True, left decreases in the same amount as right increases.
        If False, only left decreases and right is not affected.

    """
    def __init__(self, name: str, left: str, right: str, fun: Callable, symmetrical: bool = True) -> None:
        self.name = name
        self.left = left
        self.right = right
        self.fun = fun
        self.symmetrical = symmetrical


class AbstractModel(ABC):
    """An abstract class to be inherited by all models. It contains all the basic
    functionality that all models should have. It is used to describe the
    structure of a model in an interpretable way. It is also used to store
    the data and the parameters of the model.

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

    Notes
    -----
     - If you wish to create a new model as a class you should inherit from this class and define the states and
     transitions of the model.
     - If you wish to create a model and define the states and transitions in runtime
     you should use the EmptyModel class.
    """
    def __init__(self,
                 name: str,
                 hyper_params: dict,
                 train_params: dict,
                 static_params: dict,
                 *args: Any,
                 **kwargs: Any) -> None:

        self.name = name
        self.states = {}
        self.transitions = []
        self.hyper_params = hyper_params
        self.train_params = train_params
        self.static_params = static_params
        self.train_data = None
        self.test_data = None
        self.trained = False
        self.shapes = None
        self.sizes = None
        self.save_shapes_and_sizes_of_train_params()

    def set_train_data(self, train_data):
        self.train_data = train_data

    def set_test_data(self, test_data):
        self.test_data = test_data

    def get_param(self, name: str) -> Any:
        """Returns the value of a parameter by searching its name in all parameter namespaces.
        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        Any
            The value of the parameter.

        Raises
        ------
        KeyError
            If the parameter is not found.
        """
        if name in self.hyper_params.keys():
            return self.hyper_params[name]
        elif name in self.train_params.keys():
            return self.train_params[name]
        elif name in self.static_params.keys():
            return self.static_params[name]
        else:
            raise KeyError(f"Parameter with name '{name}' was not found.")

    def p(self, name: str) -> Any:
        """Shorthand of get_param."""
        return self.get_param(name)

    def add_state(self, a_state: State) -> None:
        """Adds a state to the model."""
        self.states[a_state.name] = a_state

    def add_transition(self, a_transition: Transition) -> None:
        """Adds a transition to the model."""
        self.transitions.append(a_transition)

    def add_states(self, states: list) -> None:
        """Adds a list of states to the model."""
        for state in states:
            self.add_state(state)

    def add_transitions(self, transitions: list) -> None:
        """Adds a list of transitions to the model."""
        for transition in transitions:
            self.add_transition(transition)

    def save_shapes_and_sizes_of_train_params(self) -> None:
        """
        Saves the shapes and sizes of the train parameters. It is used to reshape the parameters after training.
        Returns
        -------
            None
        """
        shapes = []
        sizes = []
        for key, value in self.train_params.items():
            # TODO: This is a temporary solution. It should be changed.
            # Using isinstance is kinda smelly really
            if isinstance(value, list):
                shapes.append(len(value))
                sizes.append(len(value))
            elif isinstance(value, float) or isinstance(value, int):
                shapes.append(1)
                sizes.append(1)
            else:
                shapes.append(value.shape)
                sizes.append(value.size)

        self.shapes = shapes
        self.sizes = sizes

    def get_train_params_as_list(self):
        """
        Returns the train parameters as a list.
        Returns
        -------
        list
            The train parameters as a list.
        """
        params = []
        for key, value in self.train_params.items():
            if isinstance(value, list):
                params += value
            elif isinstance(value, float) or isinstance(value, int):
                params.append(value)
            else:
                params += value.flatten().tolist()
        return params

    def set_train_params_from_list(self, list_of_params):
        """
        Sets the train parameters from a list. In the same order of
        get_train_params_as_list. (Order matters!!)

        Parameters
        ----------
        list_of_params: list
            The list of train parameters to be saved.

        Returns
        -------
            None
        """
        last_taken_element = 0
        for size, shape, param_name, in zip(self.sizes, self.shapes, self.train_params.keys()):
            self.train_params[param_name] = np.array(
                list_of_params[last_taken_element:last_taken_element + size]
            ).reshape(shape)
            last_taken_element += size

    # TODO: Beutify this. Fix kwargs.
    def plot_model(self, ax: Axes = None, **kwargs: Any) -> Union[Axes, None]:
        """Plots the model using the networkx library.
        Parameters
        ----------
        ax : Axes
            The axes to plot the model on. If None is given, a new figure is created.
        kwargs : Any
            The keyword arguments to be passed to the networkx draw function.
        Returns
        -------
        Union[Axes, None]
            If an ax was passed it returns it.
            If no ax was passed it returns None.
        """
        g = nx.Graph()
        edges_label = {}
        edges = []

        for t in self.transitions:
            edges.append([t.left, t.right])
            edges_label[(t.left, t.right)] = t.name

        g.add_edges_from(edges)
        pos = nx.spring_layout(g)
        nx.draw_networkx(g, pos, ax=ax, **kwargs)
        nx.draw_networkx_edge_labels(g, pos, edges_label, ax=ax, **kwargs)
        if ax is not None:
            return ax

    def __repr__(self):
        """Returns a string representation of the model."""
        string_rep = f"Model: {self.name}\n"
        string_rep += f"States: {list(self.states.keys())}\n"
        string_rep += f"Transitions: {[t.name for t in self.transitions]}\n"
        string_rep += f"Has train data? {self.train_data is not None}\n"
        string_rep += f"Has test data? {self.test_data is not None}\n"
        string_rep += f"Trained? {self.trained}\n"
        return string_rep


class EmptyModel(AbstractModel):
    """An empty model. It is used to create a model from scratch. It
    is instantiated with no states or transitions."""
    pass
