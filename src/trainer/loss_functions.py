import numpy as np

"""
Este módulo foi gerado inicialmente pelo ChatGPT e modificado manualmente
para fornecer funções de custo (loss functions) e suas derivadas para uso em redes neurais.

Cada função inclui:
- Overview: explicação breve
- Casos bons para uso
- Quando evitar

Autor: Renato Calabro
"""

# 🔹 Mean Squared Error (MSE)

def mse(y_pred, y_true):
    """
    Overview: Calcula o Erro Quadrático Médio entre previsão e valor real.
    Casos bons para uso: Regressão e saída contínua.
    Quando evitar: Problemas de classificação com saídas binárias ou probabilísticas.
    """
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    """Derivada do MSE em relação à saída da rede."""
    return 2 * (y_pred - y_true) / y_true.size


# 🔹 Binary Cross-Entropy

def binary_crossentropy(y_pred, y_true):
    """
    Overview: Mede a perda entre probabilidades previstas e classes binárias.
    Casos bons para uso: Classificação binária (0 ou 1) na saída.
    Quando evitar: Saídas contínuas ou multiclasse.
    """
    epsilon = 1e-9  # Evita log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_pred, y_true):
    """Derivada da Binary Cross-Entropy com clipping para evitar divisão por zero."""
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return ((-y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / y_true.size

# 🔹 Dicionários para uso dinâmico

LOSSES = {
    "mse": mse,
    "binary_crossentropy": binary_crossentropy,
}

LOSS_DERIVATIVES = {
    "mse": mse_derivative,
    "binary_crossentropy": binary_crossentropy_derivative,
}
