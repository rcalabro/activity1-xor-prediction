from .execution.basic_loop import BasicLoopExecution

from .training.vanilla_backpropagation import VanillaBackpropagation
# futuras: momentum, minibatch, etc

class Trainer:
    def __init__(self, exec_strategy="simple", train_strategy="standard", **options):
        self.options = options
        self.execution = self._resolve_execution(exec_strategy, **options)
        self.training = self._resolve_training(train_strategy, **options)

    def _resolve_execution(self, name, **override_options):
        opts = {**self.options, **override_options}
        if name == "basic-loop":
            return BasicLoopExecution(**opts)
        # elif name == "auto-restart":
        #     return RestartExecutionStrategy()
        # elif name == "early-stop":
        #     return EarlyStopExecutionStrategy()
        else:
            raise ValueError(f"Estratégia de execução '{name}' não reconhecida.")

    def _resolve_training(self, name, **override_options):
        opts = {**self.options, **override_options}
        if name == "vanilla-backpropagation":
            return VanillaBackpropagation(**opts)
        else:
            raise ValueError(f"Estratégia de treinamento '{name}' não reconhecida.")

    
    def train(self, nn, X, y, **override_options):
        opts = {**self.options, **override_options}
        return self.execution.execute(nn, X, y, self.training, **opts)
