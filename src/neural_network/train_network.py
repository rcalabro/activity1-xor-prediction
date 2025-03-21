import numpy as np
from .activation_functions import ACTIVATIONS, ACTIVATION_DERIVATIVES
from .loss_functions import LOSSES, LOSS_DERIVATIVES

"""
Este módulo foi gerado inicialmente pelo ChatGPT e modificado manualmente 
para implementar o treinamento de uma rede neural com Backpropagation e suporte
a múltiplas funções de custo (loss functions).

Autor: Renato Calabro
"""

def train_network(nn, X, y, epochs=1000, learning_rate=0.001, target_error=0.05, loss_function="mse"):
    """
    Treina uma rede neural `nn` utilizando Backpropagation com Gradiente Descendente.

    Parâmetros:
    - nn: Instância da classe NeuralNetwork.
    - X: Dados de entrada (shape: [amostras, neurônios de entrada]).
    - y: Saídas esperadas (shape: [amostras, neurônios de saída]).
    - epochs: Número máximo de épocas de treinamento.
    - learning_rate: Taxa de aprendizado.
    - target_error: Critério de parada baseado no erro.
    - loss_function: Nome da função de custo ("mse", "mae", "binary_crossentropy", "categorical_crossentropy").

    Retorna:
    - history: Lista com o erro por época.
    - epochs_executed: Número de épocas realizadas.
    """

    # 🔹 Valida função de ativação
    activation_name = nn.activation_func.__name__
    if activation_name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Derivada da função de ativação '{activation_name}' não encontrada.")
    activation_derivative = ACTIVATION_DERIVATIVES[activation_name]

    # 🔹 Valida função de custo
    loss_func = LOSSES.get(loss_function)
    loss_derivative = LOSS_DERIVATIVES.get(loss_function)
    if loss_func is None or loss_derivative is None:
        raise ValueError(f"Função de custo '{loss_function}' não suportada.")

    history = []

    for epoch in range(epochs):
        # ========== 1. Forward pass ==========
        activations = [X]
        inputs = []

        a = X
        for w, b in zip(nn.weights, nn.biases):
            z = np.dot(a, w) + b
            inputs.append(z)
            a = nn.activation_func(z)
            activations.append(a)

        # ========== 2. Cálculo do erro (loss) ==========
        y_pred = activations[-1]
        loss = loss_func(y_pred, y)
        history.append(loss)

        if loss <= target_error:
            print(f"\n✅ Erro alvo atingido na época {epoch+1}: {loss:.6f}")
            break

        # ========== 3. Backpropagation ==========
        deltas = [None] * len(nn.weights)
        error = loss_derivative(y_pred, y)
        deltas[-1] = error * activation_derivative(inputs[-1])

        for i in reversed(range(len(deltas) - 1)):
            delta_forward = deltas[i + 1]
            w_forward = nn.weights[i + 1]
            da_dz = activation_derivative(inputs[i])
            deltas[i] = np.dot(delta_forward, w_forward.T) * da_dz

        # ========== 4. Atualização dos pesos e bias ==========
        for i in range(len(nn.weights)):
            a_prev = activations[i]
            nn.weights[i] -= learning_rate * np.dot(a_prev.T, deltas[i])
            nn.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

        # ========== 5. Log de progresso ==========
        if epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} - Erro: {loss:.6f}")

    return history, epoch + 1
