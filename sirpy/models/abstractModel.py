import pandas as pd
from abc import ABC
from typing import Any
import state
import transition
from scipy.integrate import odeint
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class AbstractModel(ABC):
    def __init__(self,
                 name: str,
                 hyper_params: dict,
                 target_params: dict,
                 static_params: dict,
                 *args: Any,
                 **kwargs: Any) -> None:
        self.name = name
        self.initial_conditions = None
        self.states = {}
        self.transitions = []
        self.hyper_params = hyper_params
        self.target_params = target_params
        self.static_params = static_params
        self.train_data = None
        self.test_data = None

    def get_param(self, name):
        if name in self.hyper_params.keys():
            return self.hyper_params[name]
        elif name in self.target_params.keys():
            return self.target_params[name]
        elif name in self.static_params.keys():
            return self.static_params[name]
        else:
            raise ValueError(f"Parameter with name '{name}' was not found.")

    def add_state(self, aState: state) -> None:
        """Add a new state to the model."""
        self.states[aState.name] = aState
        aState.set_model(self)

    def add_transitions(self, aTransition: transition) -> None:
        """
        Adds a new transition between two states.
        :param aTransition: The transition to be added. Must be an object of a class that inherits from
        abstractTransition.
        :return: None
        """
        self.transitions.append(aTransition)
        aTransition.set_model(self)

    # Asume que y esta ordenado segun el orden de states
    def compute_gradients(self, y: list, t: float) -> list:
        gradients = []
        if len(y) != len(self.states):
            raise ValueError(f"Length mismatch between y and states. ({len(y)}-{len(self.states)})")

        for val, state in zip(y, self.states.values()):
            state.reset_gradient()
            state.set_value(val)

        for transition in self.transitions:
            transition.compute_gradients(t)

        for state in self.states.values():
            gradients.append(state.grad)

        return gradients

    def solve_ode_system(self):
        return odeint(
            self.compute_gradients,
            self.get_param("initial_condition"),
            self.get_param("time_space")
        )

    def save_new_target_params(self, x: list) -> None:
        for i, key in zip(x, self.target_params.keys()):
            self.target_params[key] = i

    def calculate_curves(self):
        return self.solve_ode_system()

    def plot_curves(self):
        return pd.DataFrame(self.solve_ode_system(), columns=list(self.states.keys())).plot()

    def plot_model(self, ax: Axes = None, **kwargs: Any) -> None:
        G = nx.Graph()
        edges_label = {}
        edges = []

        for t in self.transitions:
            edges.append([t.left, t.right])
            edges_label[(t.left, t.right)] = t.name

        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        nx.draw_networkx(
            G, pos, ax=ax, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in G.nodes()},
        )
        nx.draw_networkx_edge_labels(
            G, pos, edges_label, ax=ax
        )

    def __repr__(self):
        return self.name
