import signal
import sys
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, plot_network, train_network

def handle_exit(sig, frame):
    print(f"\n🛑 Sinal {sig} recebido. Fechando plots e encerrando...")
    plt.close('all')
    sys.exit(0)

def main():
    input_size = 2
    output_classes = [0, 1]

    nn = NeuralNetwork(
        input_layer=input_size,
        hidden_layers=[2],
        output_layer=1,
        activation="sigmoid",
    )


    # Tabela verdade XOR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Treino
    history = train_network(nn, X, y, epochs=100000, learning_rate=0.1, target_error=0.01, loss_function="binary_crossentropy")

    # Test
    print("\n🔹 Teste da rede neural após treinamento:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]} → Saída prevista: {nn.predict(X[i].reshape(1, -1))}, saída esperada: {y[i]}")


    # Predição
    casos = [
        [np.array([[0, 0]]), 0],
        [np.array([[0, 1]]), 1],
        [np.array([[1, 0]]), 1],
        [np.array([[1, 1]]), 0]
    ]

    def classify(pred):
        return (pred[0][0] > 0.5).astype(int)

    plot = False

    if not plot:
        print("\nPREDIÇÕES:")

    for input, expected in casos:
        pred = classify(nn.predict(input))

        if plot:
            plot_network(nn, input, show=False, width=600, height=400, title=f"XOR: {input[0].flatten()} -> {pred}")
        else:
            print(f"XOR: {input.reshape(1, -1)} -> {pred} {'✅' if pred == expected else '❌'} expected: {expected}")

    if plot:
        plt.show()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # kill <pid>
    main()
