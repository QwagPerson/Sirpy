from state import State
from transition import SimpleTransition
from abstractModel import AbstractModel
from typing import Any


class SIR(AbstractModel):
    def __init__(self, name: str, hyper_params: dict, target_params: dict, static_params: dict, *args: Any,
                 **kwargs: Any) -> None:
        # Let's create the states
        super().__init__(name, hyper_params, target_params, static_params, *args, **kwargs)

        self.add_state(State("S"))
        self.add_state(State("I"))
        self.add_state(State("R"))

        # Now the transitions
        self.add_transitions(
            SimpleTransition(
                r"$\beta SI$",
                "S",
                "I",
                lambda _: self.get_param("beta") * self.states["S"].val * self.states["I"].val
            )
        )

        self.add_transitions(
            SimpleTransition(
                r"$\gamma I$",
                "I",
                "R",
                lambda _: self.get_param("gamma") * self.states["I"].val
            )
        )
