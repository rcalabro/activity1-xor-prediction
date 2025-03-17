from neural_network import NeuralNetwork, plot_network
import numpy as np

def main():
    input_size = 2

    nn = NeuralNetwork(
        inputLayer=input_size,
        hiddenLayers=[4,4],
        outputLayer=2,
        activation="sigmoid"
    )

    X = np.random.rand(1, input_size)

    print("INPUT: ", X )

    # Forward pass
    output = nn.predict(X)
    print("Saída da rede (forward):")
    print(output)

    # Exemplo: mostrar até 10 neurônios na entrada, 8 nas ocultas e todos na saída
    plot_network(
        nn,
        X,
        width=1200,
        height=800,
        max_show_input=10,
        max_show_hidden=16,
        max_show_output=None  # None => mostra todos na camada de saída
    )

if __name__ == "__main__":
    main()
