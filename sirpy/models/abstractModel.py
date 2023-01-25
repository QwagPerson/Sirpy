from abc import ABC
from typing import Any, Callable

import networkx as nx
import numpy as np
from matplotlib.axes import Axes


class State:
    def __init__(self, name: str) -> None:
        self.name = name


class Transition:
    def __init__(self, name: str, left: str, right: str, fun: Callable, symmetrical: bool = True) -> None:
        self.name = name
        self.left = left
        self.right = right
        self.fun = fun
        self.symmetrical = symmetrical


class AbstractModel(ABC):
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

    def get_param(self, name):
        if name in self.hyper_params.keys():
            return self.hyper_params[name]
        elif name in self.train_params.keys():
            return self.train_params[name]
        elif name in self.static_params.keys():
            return self.static_params[name]
        else:
            raise KeyError(f"Parameter with name '{name}' was not found.")

    def p(self, name):
        return self.get_param(name)

    def add_state(self, a_state: State) -> None:
        self.states[a_state.name] = a_state

    def add_transition(self, a_transition: Transition) -> None:
        self.transitions.append(a_transition)

    def add_states(self, states: list) -> None:
        for state in states:
            self.add_state(state)

    def add_transitions(self, transitions: list) -> None:
        for transition in transitions:
            self.add_transition(transition)

    def save_shapes_and_sizes_of_train_params(self):
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
        last_taken_element = 0
        for size, shape, param_name, in zip(self.sizes, self.shapes, self.train_params.keys()):
            self.train_params[param_name] = np.array(
                list_of_params[last_taken_element:last_taken_element + size]
            ).reshape(shape)
            last_taken_element += size

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


class EmptyModel(AbstractModel):
    pass
