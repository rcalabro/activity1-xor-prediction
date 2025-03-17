# main.py
from neural_network import NeuralNetwork, plot_network
import numpy as np

def main():
    # Cria e usa a rede neural
    nn = NeuralNetwork(
        inputLayer=2,
        hiddenLayers=[3],
        outputLayer=1,
        activation="sigmoid"
    )

    # Exemplo de dados
    X = np.array([[0.5, 0.2]])

    # Desenha e (opcionalmente) salva o gráfico
    plot_network(nn, X, width=800, height=600)

    # Forward pass
    output = nn.predict(X)
    print("Saída da rede (forward):")
    print(output)


if __name__ == "__main__":
    main()
