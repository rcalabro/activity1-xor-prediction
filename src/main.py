from neural_network import NeuralNetwork, plot_network, train_network
import numpy as np

def main():
    input_size = 2
    output_classes = [0, 1]

    nn = NeuralNetwork(
        input_layer=input_size,
        hidden_layers=[2],
        output_layer=len(output_classes),
        activation="sigmoid",
    )


    # 🔹 Criando um dataset simples (XOR)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [1 ,0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    # 🔹 Treinar a rede (somente aleatório para testes)
    history = train_network(nn, X, y, epochs=100000, learning_rate=0.1, target_error=0.01, loss_function="binary_crossentropy")

    # 🔹 Testando a rede após o treinamento
    print("\n🔹 Teste da rede neural após treinamento:")
    for i in range(len(X)):
        print(f"Entrada: {X[i]} → Saída prevista: {nn.predict(X[i].reshape(1, -1))}, saída esperada: {y[i]}")

    # X = np.random.rand(1, input_size)
    #
    # print("INPUT: ", X )
    #
    # # Forward pass
    # output = nn.predict(X)
    # print("Saída da rede (forward):")
    # print(output)
    #
    #
    # plot_network(
    #     nn,
    #     X,
    #     width=1200,
    #     height=800,
    #     max_show_input=10,
    #     max_show_hidden=16,
    #     max_show_output=None  # None => mostra todos na camada de saída
    # )

if __name__ == "__main__":
    main()
