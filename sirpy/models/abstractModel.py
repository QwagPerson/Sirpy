import pandas as pd
from abc import ABC
from typing import Any
import state
import transition
import networkx as nx
from matplotlib.axes import Axes
from sirpy.parameter.SimpleParameter import SimpleParameter


class AbstractModel(ABC):
    def __init__(self,
                 name: str,
                 hyper_params: dict,
                 train_params: dict,
                 static_params: dict,
                 *args: Any,
                 **kwargs: Any) -> None:
        self.name = name
        self.initial_conditions = None
        self.states = {}
        self.transitions = []
        self.hyper_params = hyper_params
        self.train_params = train_params
        self.static_params = static_params
        self.train_data = None
        self.test_data = None
        self.trained = False

    def make_parameters_object(self):
        for param in list(self.train_params.values())+list(self.static_params.values()):
            if param is float:
                param = SimpleParameter(param, self.train_params[param])

    def get_param(self, name):
        if name in self.hyper_params.keys():
            return self.hyper_params[name]
        elif name in self.train_params.keys():
            return self.train_params[name]
        elif name in self.static_params.keys():
            return self.static_params[name]
        else:
            raise ValueError(f"Parameter with name '{name}' was not found.")

    def add_state(self, a_state: state) -> None:
        """Add a new state to the model."""
        self.states[a_state.name] = a_state
        a_state.set_model(self)

    def add_transitions(self, a_transition: transition) -> None:
        """
        Adds a new transition between two states.
        :param: a_transition: The transition to be added. Must be an object of a class that inherits from
        abstractTransition.
        :return: None
        """
        self.transitions.append(a_transition)
        a_transition.set_model(self)

    # TODO: This is a temporary solution. It should be changed.
    # It should support n-dimensional parameters and distributions as well.
    def save_new_target_params(self, x: list) -> None:
        for i, key in zip(x, self.train_params.keys()):
            self.train_params[key] = i

    # TODO: Beutify this. Fix kwargs.
    def plot_model(self, ax: Axes = None, **kwargs: Any) -> None:
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

    def __repr__(self):
        string_rep = f"Model: {self.name}\n"
        string_rep += f"States: {list(self.states.keys())}\n"
        string_rep += f"Transitions: {[t.name for t in self.transitions]}\n"
        string_rep += f"Has train data? {self.train_data is not None}\n"
        string_rep += f"Has test data? {self.test_data is not None}\n"
        string_rep += f"Trained? {self.trained}\n"
        return string_rep
