from abstractModel import AbstractModel, State, Transition
from typing import Any


class SIR(AbstractModel):
    def __init__(self, name: str, hyper_params: dict, train_params: dict, static_params: dict, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(name, hyper_params, train_params, static_params, *args, **kwargs)

        # Let's create the states
        self.add_states([State("S"), State("I"), State("R")])

        # Now the transitions
        self.add_transition(
            Transition(r"$\beta SI$", "S", "I",
                       lambda y, _: self.p("beta") * y["S"] * y["I"]
                       )
        )

        self.add_transition(
            Transition(r"$\gamma I$", "I", "R",
                       lambda y, _: self.p("gamma") * y["I"]
                       )
        )
