import matplotlib.pyplot as plt

# Dados de treino
x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 6, 8, 10]

# Parâmetros do modelo
w = 0.5
lr = 0.01
epochs = 100
batch_size = 2

# Armazenar perdas para plotar depois
loss_history = []

# Função de perda (MSE)
def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

# Derivada do erro em relação ao peso
def grad(x, y_pred, y_true):
    return 2 * (y_pred - y_true) * x

# Treinamento
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(x_data), batch_size):
        batch_x = x_data[i:i + batch_size]
        batch_y = y_data[i:i + batch_size]

        # Inicializa o gradiente acumulado
        total_grad = 0
        for x, y in zip(batch_x, batch_y):
            y_pred = w * x
            epoch_loss += loss(y_pred, y)
            total_grad += grad(x, y_pred, y)

        # Atualiza o peso com o gradiente médio do batch
        w = w - lr * (total_grad / len(batch_x))

    # Armazena a média da perda da epoch
    loss_history.append(epoch_loss / len(x_data))

# Resultado final
print(f"\nPeso final aprendido: w = {w:.4f}")

# 📊 Gráfico da perda ao longo das epochs
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (Erro Médio)')
plt.title('Treinamento - Redução do Erro ao Longo das Epochs')
plt.grid(True)
plt.show()
