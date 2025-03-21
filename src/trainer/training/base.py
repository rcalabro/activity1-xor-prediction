from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    def __init__(self, **options):
        self.learning_rate = options.get("learning_rate")
        self.loss_function = options.get("loss_function")
        self.verbose = options.get("verbose", False)
        self.name = 'base'

    @abstractmethod
    def train_step(self, nn, X, y, **options):
        """
        Executa uma única etapa de atualização de pesos e biases.

        Parâmetros:
        - nn: instância da rede neural
        - X: batch de entrada
        - y: batch de saída
        - options: opções específicas para a estratégia

        Retorna:
        - loss: erro médio da etapa
        """
        pass
