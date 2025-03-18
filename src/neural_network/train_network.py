import numpy as np

"""
Este módulo foi gerado inicialmente pelo ChatGPT e posteriormente modificado manualmente 
para incluir um algoritmo de treinamento experimental.

Implementa um treinamento **aleatório** para testes, sem utilizar Gradiente Descendente.
Os pesos e bias da rede são atualizados aleatoriamente para minimizar uma função de custo.

Autor: Renato Calabro
"""

def train_network(nn, X, y, epochs=100, learning_rate=0.1, target_error=0.01):
    """
    Treina uma rede neural `nn` ajustando pesos e bias de forma aleatória.

    Parâmetros:
    - nn: Instância de NeuralNetwork já inicializada.
    - X: Dados de entrada (shape: [amostras, neurônios de entrada]).
    - y: Dados de saída esperados (shape: [amostras, neurônios de saída]).
    - epochs: Número máximo de épocas de treinamento (padrão: 100).
    - learning_rate: Taxa de aprendizado para ajustar os pesos aleatoriamente (padrão: 0.1).
    - target_error: Valor limite para o erro, onde o treinamento será interrompido se atingido (padrão: 0.01).

    Retorna:
    - Histórico de erro em cada época.
    - Número de épocas executadas.
    """

    history = []  # Para armazenar a função de custo em cada época

    for epoch in range(epochs):
        # 🔹 Forward pass: Faz a predição da rede
        y_pred = nn.predict(X)

        # 🔹 Calcula a função de custo (Erro Quadrático Médio - MSE)
        loss = np.mean((y_pred - y) ** 2)
        history.append(loss)

        # 🔹 Verifica se atingiu o erro alvo
        if loss <= target_error:
            print(f"\n✅ Treinamento encerrado na época {epoch+1}: Erro atingiu {loss:.6f} (meta: {target_error})")
            break

        # 🔹 Atualiza pesos e bias aleatoriamente (apenas para testes)
        for i in range(len(nn.weights)):
            nn.weights[i] += np.random.randn(*nn.weights[i].shape) * learning_rate
            nn.biases[i] += np.random.randn(*nn.biases[i].shape) * learning_rate

        # 🔹 Exibe o progresso do treinamento a cada 10% das épocas ou na última
        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} - Erro: {loss:.6f}")

    return history, epoch + 1  # Retorna o histórico e o número real de épocas executadas
