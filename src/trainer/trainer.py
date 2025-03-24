import numpy as np

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

    def _save_checkpoint(self, nn, checkpoint_path):
        """
        Salva os parâmetros da rede neural treinada em um arquivo .npz
        """
        weights = np.empty(len(nn.weights), dtype=object)
        biases = np.empty(len(nn.biases), dtype=object)
        for i in range(len(nn.weights)):
            weights[i] = nn.weights[i]
            biases[i] = nn.biases[i]

        np.savez(checkpoint_path,
                 input_layer=int(nn.input_layer),
                 hidden_layers=np.array(list(map(int, nn.hidden_layers))),
                 output_layer=int(nn.output_layer),
                 activation_func=nn.activation_func.__name__,
                 weights=weights,
                 biases=biases)

    def train(self, nn, X, y, **override_options):
        opts = {**self.options, **override_options}
        history, epochs, success = self.execution.execute(nn, X, y, self.training, **opts)

        # 🔸 Salvamento de checkpoint, se solicitado e o treino teve sucesso
        checkpoint_path = opts.get("save_checkpoint", None)
        if success and checkpoint_path:
            self._save_checkpoint(nn, checkpoint_path)

        return history, epochs, success
