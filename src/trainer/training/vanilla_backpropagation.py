import numpy as np
from ..loss_functions import LOSSES, LOSS_DERIVATIVES
from .base import TrainingStrategy

"""
Esta estratégia implementa o algoritmo de treinamento padrão via backpropagation
com gradiente descendente. Utiliza a derivada da função de ativação para propagar
os erros e atualizar os pesos/bias.

Autor: Renato Calabro
"""

class VanillaBackpropagation(TrainingStrategy):
    def __init__(self, **options):
        super().__init__(**options)
        self.name = "vanilla-backpropagation"

    def train_step(self, nn, X, y, **options):
        """
        Executa uma única etapa de backpropagation com ajuste dos pesos.

        Retorna:
        - loss: erro médio da predição atual
        """

        # 🔹 Recupera a função de custo e sua derivada
        loss_func = LOSSES.get(self.loss_function)
        loss_derivative = LOSS_DERIVATIVES.get(self.loss_function)
        if loss_func is None or loss_derivative is None:
            raise ValueError(f"Função de custo '{self.loss_function}' não suportada.")

        # 🔹 Forward pass: cálculo das ativações
        activations = [X]
        inputs = []
        a = X
        for w, b in zip(nn.weights, nn.biases):
            z = np.dot(a, w) + b
            inputs.append(z)
            a = nn.activation_func(z)
            activations.append(a)

        # 🔹 Cálculo da perda (loss)
        y_pred = activations[-1]
        loss = loss_func(y_pred, y)

        # 🔹 Backpropagation: cálculo dos deltas (gradientes)
        deltas = [None] * len(nn.weights)
        error = loss_derivative(y_pred, y)
        deltas[-1] = error * nn.activation_derivative(inputs[-1])

        for i in reversed(range(len(deltas) - 1)):
            delta_forward = deltas[i + 1]
            w_forward = nn.weights[i + 1]
            da_dz = nn.activation_derivative(inputs[i])
            deltas[i] = np.dot(delta_forward, w_forward.T) * da_dz


        # 🔹 Atualização dos pesos e bias
        for i in range(len(nn.weights)):
            a_prev = activations[i]
            nn.weights[i] -= self.learning_rate * np.dot(a_prev.T, deltas[i])
            nn.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

        return loss
