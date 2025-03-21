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

# 🔹 Dicionários para uso dinâmico

LOSSES = {
    "mse": mse,
}

LOSS_DERIVATIVES = {
    "mse": mse_derivative,
}
