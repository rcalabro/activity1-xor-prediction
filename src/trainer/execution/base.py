import time
from abc import ABC, abstractmethod

class ExecutionStrategy(ABC):
    def __init__(self, **options):
        self.target_error = options.get("target_error")
        self.verbose = options.get("verbose", False)
        self.name = 'base'
        self.epochs = options.get("epochs")

    def _log_epoch(self, epoch, loss):
        print(f"Epoch {epoch+1}/{self.epochs}: {loss:.6f}")


    def execute(self, nn, X, y, training_strategy, **options):
        if self.verbose:
            print("🔹 Iniciando Treinamento")
            print(f"    Execução: {self.name}")
            print(f"    Stratégia: {training_strategy.name}")
            print(f"    Loss: {options.get('loss_function')}")
            print(f"    Learning Rate: {options.get('learning_rate')}")
            print(f"    Epochs: {options.get('epochs')}")
            print()
        else:
            print(f"🔹 Training: epochs={options.get('epochs')} | strategy={training_strategy.name}")

        start = time.perf_counter()
        history, epoch, success = self._execute(nn, X, y, training_strategy, **options)
        end = time.perf_counter()

        if success:
            print(f"\n✅ Erro alvo atingido na época {epoch+1} - error: {history[-1]:.6f}")
        else:
            print(f"\n❌ Target NOT met -> epoch: {epoch} - error: {history[-1]:.6f}")

        print(f">>> Treinamento Finalizado - ⏱️: {end - start:.4f} segundos\n")
        return history, epoch

    @abstractmethod
    def _execute(self, nn, X, y, training_strategy, **options):
        """
        Executa o ciclo de treinamento com a estratégia fornecida.

        Parâmetros:
        - nn: instância da rede neural ou função fábrica
        - X: entradas
        - y: saídas esperadas
        - training_strategy: instância de TrainingStrategy
        - options: parâmetros adicionais (ex: epochs, learning_rate)

        Retorna:
        - history: lista de erros
        - epochs_executed: total de épocas
        """
        pass
