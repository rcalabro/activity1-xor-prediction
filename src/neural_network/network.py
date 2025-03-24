import time
import numpy as np
from .activation_functions import ACTIVATIONS, ACTIVATION_DERIVATIVES

"""
Implementação baseada no livro "Make Your Own Neural Network" de Tariq Rashid, 
com adaptações específicas para atender às necessidades do projeto.

- Ajustes na inicialização dos pesos e bias.
- Uso de funções de ativação configuráveis.
- Estrutura modular para facilitar expansão e manutenção.
- Melhorias na legibilidade e organização do código.
- Validação dos pesos e bias fornecidos.

Alterações foram feitas manualmente conforme necessidade, utilizando pontualmente 
o ChatGPT para otimização e refinamento de trechos específicos.

Autor: Renato Calabro
"""


class NeuralNetwork:
    def __init__(self,
                 input_layer,
                 hidden_layers,
                 output_layer,
                 activation="sigmoid",
                 output_classification=None,
                 weights=None,
                 biases=None,
                 verbose=False):
        """
        Construtor da classe NeuralNetwork.

        Parâmetros:
        - input_layer: Número de neurônios na camada de entrada.
        - hidden_layers: Lista contendo o número de neurônios em cada camada oculta. Ex: [6, 6].
        - output_layer: Número de neurônios na camada de saída.
        - activation: Tipo de função de ativação.
        - weights: Lista opcional de matrizes numpy para os pesos (se None, inicializa aleatório).
        - biases: Lista opcional de vetores numpy para os biases (se None, inicializa aleatório).
        """

        if verbose:
            print("🔹 Inicializando Rde Neural")
            print(f"    Camada de entrada: {input_layer} neurônios")
            print(f"    Camadas ocultas: {hidden_layers}")
            print(f"    Camada de saída: {output_layer} neurônios")
            print(f"    Função de ativação: {activation}")
            print(f"    Pesos iniciais: {'CHECKPOINT' if not weights is None else 'RANDOM'}")
            print(f"    Bias iniciais: {'CHECKPOINT' if not biases is None else 'RANDOM'}\n")
        else:
            print(f"🔹 NeuralNetwork: [{input_layer}, {hidden_layers}, {output_layer}] | activation: {activation}\n")

        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.output_classification = output_classification
        self.verbose = verbose

        # Verifica se a função de ativação requisitada está no dicionário.
        if activation not in ACTIVATIONS:
            raise ValueError(f"Função de ativação '{activation}' não suportada. "
                             f"Escolha entre: {list(ACTIVATIONS.keys())}")
        self.activation_func = ACTIVATIONS[activation]
        if activation not in ACTIVATION_DERIVATIVES:
            raise ValueError(f"Função de ativação '{activation}' não suportada não ter derivada implementada")

        # para treinamento
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

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
            if weights is not None and biases is not None: # Se foram fornecidos, verifica o formato e usa os valores passados
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

            if verbose:
                print(f"--- Camada {i+1} --> neurons: {layer_sizes[i]} -> {layer_sizes[i+1]}")

        if verbose:
            print("\n>>> Rede Neural Inicializada\n")

    @staticmethod
    def load_checkpoint(path, output_classification=None, verbose=False):
        """
        Carrega uma rede neural salva em checkpoint (.npz) e retorna a instância pronta.

        Parâmetros:
        - path: caminho para o arquivo .npz gerado com save_checkpoint
        - verbose: se deve imprimir os logs de inicialização
        - output_classification: função opcional de classificação final (ex: lambda x: x > 0.5)

        Retorna:
        - Instância de NeuralNetwork pronta para uso
        """
        data = np.load(path, allow_pickle=True)

        input_layer = int(data["input_layer"])
        hidden_layers = data["hidden_layers"].tolist()
        output_layer = int(data["output_layer"])
        activation = str(data["activation_func"])
        weights = data["weights"]
        biases = data["biases"]

        return NeuralNetwork(
            input_layer=input_layer,
            hidden_layers=hidden_layers,
            output_layer=output_layer,
            activation=activation,
            weights=weights,
            biases=biases,
            verbose=verbose,
            output_classification=output_classification
        )

    def predict(self, a):
        """
        Executa o forward pass pela rede e retorna a saída final.
        """

        start = time.perf_counter()
        X = np.array(a)
        for i, (w, b) in enumerate(zip(self.weights, self.biases), start=1):
            z = np.dot(X, w) + b  # Multiplicação matricial + bias
            X = self.activation_func(z)  # Aplica a função de ativação selecionada

        end = time.perf_counter()


        if self.verbose:
            print(f"🔹 Forward pass {a} - ⏱️: {(end - start)*1000:.6f}ms")
            print(f"    --> prediction: - {X.flatten()}")
        if self.output_classification is not None:
            classified = self.output_classification(X)
            if self.verbose:
                print(f"    --> classified: - {classified}")
            return self.output_classification(X)
        return X
