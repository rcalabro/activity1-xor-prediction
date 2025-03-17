from neural_network import NeuralNetwork, plot_network
import numpy as np

def main():
    input_size = 2

    # Pesos e bias personalizados com os tamanhos corretos
    # 🔹 Definição manual de pesos e bias (listas normais)
    custom_weights = [
        np.array([
            [0.5, -0.2, 0.3],  # Conexões da camada de entrada → oculta (3 neurônios)
            [0.1,  0.4, -0.5]
        ]),

        np.array([
            [0.7, 0.3],  # Conexões da camada oculta → saída (1 neurônio)
            [-0.3, 0.5],
            [0.2, 0.01]
        ])
    ]

    custom_biases = [
        np.array([
            [0.1, -0.1, 0.05]
        ]),  # Bias para camada oculta (3 neurônios)
        np.array([
            [0.2, 0.5],
        ])               # Bias para camada de saída (1 neurônio)
    ]

    nn = NeuralNetwork(
        input_layer=2,
        hidden_layers=[3],
        output_layer=2,
        activation="sigmoid",
        weights=custom_weights,
        biases=custom_biases
    )

    X = np.array([[1,0]])

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
