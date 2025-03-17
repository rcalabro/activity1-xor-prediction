import numpy as np
from scipy.special import expit

"""
Implementação baseada no livro "Make Your Own Neural Network" de Tariq Rashid, 
com adaptações específicas para atender às necessidades do projeto.

- Ajustes na inicialização dos pesos e bias.
- Uso de funções de ativação configuráveis (sigmoid e ReLU).
- Estrutura modular para facilitar expansão e manutenção.
- Melhorias na legibilidade e organização do código.
- Validação dos pesos e bias fornecidos.

Alterações foram feitas manualmente conforme necessidade, utilizando pontualmente 
o ChatGPT para otimização e refinamento de trechos específicos.

Autor: Renato Calabro
"""

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
                 input_layer,
                 hidden_layers,
                 output_layer,
                 activation="sigmoid",
                 learning_rate=0.1,
                 weights=None,
                 biases=None):
        """
        Construtor da classe NeuralNetwork.

        Parâmetros:
        - input_layer: Número de neurônios na camada de entrada.
        - hidden_layers: Lista contendo o número de neurônios em cada camada oculta. Ex: [6, 6].
        - output_layer: Número de neurônios na camada de saída.
        - activation: Tipo de função de ativação ("sigmoid" ou "relu").
        - learning_rate: Taxa de aprendizado (padrão=0.1).
        - weights: Lista opcional de matrizes numpy para os pesos (se None, inicializa aleatório).
        - biases: Lista opcional de vetores numpy para os biases (se None, inicializa aleatório).
        """

        print("=== Inicializando Rede Neural ===")
        print(f"- Camada de entrada: {input_layer} neurônios")
        print(f"- Camadas ocultas: {hidden_layers}")
        print(f"- Camada de saída: {output_layer} neurônios")
        print(f"- Função de ativação: {activation}")
        print(f"- Taxa de aprendizado: {learning_rate}")
        print(f"- Pesos pré-definidos: {'Sim' if weights else 'Não'}")
        print(f"- Bias pré-definidos: {'Sim' if biases else 'Não'}\n")

        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = learning_rate

        # Verifica se a função de ativação requisitada está no dicionário.
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não suportada. "
                             f"Escolha entre: {list(ACTIVATIONS.keys())}")
        self.activation_func = ACTIVATIONS[activation]

        # Lista de pesos e biases
        self.weights = []
        self.biases = []

        # Definir tamanho das camadas
        layer_sizes = [self.input_layer] + self.hidden_layers + [self.output_layer]

        # Validar pesos e bias fornecidos
        if weights is not None and biases is not None:
            if len(weights) != len(layer_sizes) - 1:
                raise ValueError(f"Esperado {len(layer_sizes) - 1} matrizes de pesos, "
                                 f"mas foram recebidas {len(weights)}.")

            if len(biases) != len(layer_sizes) - 1:
                raise ValueError(f"Esperado {len(layer_sizes) - 1} vetores de bias, "
                                 f"mas foram recebidos {len(biases)}.")

        # Inicializar pesos e bias (predefinidos ou aleatórios)
        for i in range(len(layer_sizes) - 1):
            if weights and biases:  # Se foram fornecidos, verifica o formato e usa os valores passados
                if weights[i].shape != (layer_sizes[i], layer_sizes[i+1]):
                    raise ValueError(f"Formato inválido para pesos na camada {i+1}: "
                                     f"Esperado ({layer_sizes[i]}, {layer_sizes[i+1]}), "
                                     f"recebido {weights[i].shape}.")

                if biases[i].shape != (1, layer_sizes[i+1]):
                    raise ValueError(f"Formato inválido para bias na camada {i+1}: "
                                     f"Esperado (1, {layer_sizes[i+1]}), "
                                     f"recebido {biases[i].shape}.")

                w = weights[i]
                b = biases[i]
            else:  # Se não, inicializa aleatoriamente
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1])
                b = np.random.randn(1, layer_sizes[i+1])

            self.weights.append(w)
            self.biases.append(b)

            print(f"--- Camada {i+1} ---")
            print(f"   Número de neurônios: {layer_sizes[i]} -> {layer_sizes[i+1]}")
            print(f"   Pesos shape: {w.shape}, Bias shape: {b.shape}")
            print(f"   Pesos (primeiros 5 valores): {w.flatten()[:5]}")
            print(f"   Bias (primeiros 5 valores): {b.flatten()[:5]}")
            print()

        print("=== Fim da inicialização ===\n")

    def predict(self, a):
        """
        Executa o forward pass pela rede e retorna a saída final.
        """
        print(">>> Iniciando forward pass...")
        for i, (w, b) in enumerate(zip(self.weights, self.biases), start=1):
            z = np.dot(a, w) + b  # Multiplicação matricial + bias
            a = self.activation_func(z)  # Aplica a função de ativação selecionada
            print(f"   - Camada {i}: Z shape {z.shape}, A shape {a.shape}")
        print(">>> Forward pass finalizado.\n")
        return a
