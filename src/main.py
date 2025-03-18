from neural_network import NeuralNetwork, plot_network
import numpy as np

def main():
    input_size = 2
    output_layer = [0,1]

    nn = NeuralNetwork(
        input_layer=input_size,
        hidden_layers=[3],
        output_layer=len(output_layer),
        activation="sigmoid",
    )

    X = np.random.rand(1, input_size)

    print("INPUT: ", X )

    # Forward pass
    output = nn.predict(X)
    print("Saída da rede (forward):")
    print(output)


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
