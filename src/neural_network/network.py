import numpy as np
from scipy.special import expit

def sigmoid(x):
    """Função de ativação Sigmoid."""
    return expit(x)

def relu(x):
    """Função de ativação ReLU."""
    return np.maximum(0, x)

# Dicionário com funções de ativação suportadas.
ACTIVATIONS = {
    "sigmoid": sigmoid,
    "relu": relu
}

class NeuralNetwork:
    def __init__(self,
                 inputLayer,
                 hiddenLayers,
                 outputLayer,
                 activation="sigmoid",
                 learningRate=0.1):
        """
        Construtor da classe NeuralNetwork.

        Parâmetros:
        - inputLayer: Número de neurônios na camada de entrada.
        - hiddenLayers: Lista contendo o número de neurônios em cada camada oculta. Ex: [6, 6].
        - outputLayer: Número de neurônios na camada de saída.
        - activation: Tipo de função de ativação ("sigmoid" ou "relu").
        - learningRate: Taxa de aprendizado (padrão=0.1).
        """

        print("=== Inicializando Rede Neural ===")
        print(f"- Camada de entrada: {inputLayer} neurônios")
        print(f"- Camadas ocultas: {hiddenLayers}")
        print(f"- Camada de saída: {outputLayer} neurônios")
        print(f"- Função de ativação escolhida: {activation}")
        print(f"- Taxa de aprendizado: {learningRate}\n")

        self.inputLayer = inputLayer
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer
        self.learningRate = learningRate

        # Verifica se a função de ativação requisitada está no dicionário.
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não suportada. "
                             f"Escolha entre: {list(ACTIVATIONS.keys())}")
        self.activation_func = ACTIVATIONS[activation]

        # Listas para armazenar os pesos e biases de cada camada.
        self.weights = []
        self.biases = []

        # Montamos uma lista com todas as camadas: entrada, ocultas e saída.
        layer_sizes = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]

        # Inicialização aleatória dos pesos e biases.
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            b = np.random.randn(1, layer_sizes[i+1])

            self.weights.append(w)
            self.biases.append(b)

            print(f"--- Camada {i+1} ---")
            print(f"   Número de neurônios: {layer_sizes[i]} -> {layer_sizes[i+1]}")
            print(f"   Pesos iniciais shape: {w.shape}, Bias shape: {b.shape}")
            print(f"   Pesos iniciais (primeiros 5 valores): {w.flatten()[:5]}")
            print(f"   Bias iniciais (primeiros 5 valores): {b.flatten()[:5]}")
            print()

        print("=== Fim da inicialização ===\n")

    def predict(self, A):
        """
        Executa o forward pass pela rede e retorna a saída final.
        """
        print(">>> Iniciando forward pass...")
        for i, (w, b) in enumerate(zip(self.weights, self.biases), start=1):
            Z = np.dot(A, w) + b  # Multiplicação matricial + bias
            A = self.activation_func(Z)  # Aplica a função de ativação selecionada
            print(f"   - Camada {i}: Z shape {Z.shape}, A shape {A.shape}")
        print(">>> Forward pass finalizado.\n")
        return A


