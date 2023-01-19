from sirpy.parameter.AbstractParameter import AbstractParameter


class SimpleParameter(AbstractParameter):
    def __init__(self, name: str, value: float):
        super().__init__(name, value)

    def add_to_bayesian_model(self, bayes_trainer):
        bayes_trainer.add_deterministic_param(self.name, self.value, **self.kwargs)