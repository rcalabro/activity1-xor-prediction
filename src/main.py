import signal
import sys
import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork, plot_network, train_network

def xor_classification(pred):
    def toClass(value):
        return (value > 0.5).astype(int)

    return np.array([toClass(x[0]) for x in pred])

def create_xor_nn(verbose=False):
    return NeuralNetwork(
        input_layer=2, # dois neuronios na camada de entrada
        hidden_layers=[2], # menor rede testada com sucesso
        output_layer=1, # 1 neuronio de output
        activation="sigmoid",
        verbose=verbose,
        output_classification=xor_classification,
    )


def train_xor(nn, epochs=100000, target_error=0.24, learning_rate=0.5, verbose=False):
    # Dataset -> Tabela verdade XOR
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    y = [
        [0],
        [1],
        [1],
        [0]
    ]

    history = train_network(nn,
                            np.array(X),
                            np.array(y),
                            epochs=epochs,
                            learning_rate=learning_rate,
                            target_error=target_error,
                            loss_function="binary_crossentropy",
                            verbose=verbose)
    return history


def main():
    plot = False
    verbose = False
    epochs = 1000

    xor_nn = create_xor_nn(verbose=verbose)
    train_xor(xor_nn, epochs, target_error=0.25, learning_rate=1, verbose=verbose)

    # pode ser o dataset de treino repetido aqui pelo caso de uso XOR ser determinado e pequeno
    # apenas simulando como se fossem outros casos para teste
    cases = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    expected = [
        0,
        1,
        1,
        0
    ]

    results = xor_nn.predict(cases)
    for case, pred, expected in zip(cases, results, expected):
        if plot:
            plot_network(xor_nn, case, show=False, show_labels=True, width=600, height=400, title=f"XOR: {case} -> {pred} expected: {expected}")
        print(f"XOR: {case} -> {pred} {'✅' if pred == expected else '❌'} expected: {expected}")

    if plot:
        plt.show()


if __name__ == "__main__":
    def handle_exit(sig, frame):
        print(f"\n🛑 {sig} -> shutting down...")
        plt.close('all')
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # kill <pid>
    main()
