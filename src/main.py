import signal
import sys
import numpy as np
import matplotlib.pyplot as plt

from trainer import Trainer
from neural_network import NeuralNetwork, plot_network

def create_xor_nn(verbose=False):
    def xor_classification(pred):
        def toClass(value):
            return (value > 0.5).astype(int)

        return np.array([toClass(x[0]) for x in pred])

    return NeuralNetwork(
        input_layer=2, # dois neuronios na camada de entrada
        hidden_layers=[2], # menor rede testada com sucesso
        output_layer=1, # 1 neuronio de output
        activation="sigmoid",
        verbose=verbose,
        output_classification=xor_classification,
    )


def train_xor(nn, X, y, epochs, target_error, learning_rate, verbose=False):
    trainer = Trainer(
        exec_strategy="simple",            # ou "simple", "auto-restart"
        train_strategy="standard",         # outras podem ser plugadas depois
        learning_rate=learning_rate,
        loss_function="binary_crossentropy",
        target_error=target_error,
        epochs=epochs,
        verbose=True
    )

    history, epochs = trainer.train(nn, np.array(X), np.array(y))
    return history, epochs


def main():
    plot = False

    # para fins do exercicio o dataset de treino e teste serão os mesmo ja que os casos de XOR são 4 entradas diferentes somente
    X_train = X_test    =    [[0, 0],[0, 1],[1, 0],[1, 1]]
    y_train = y_test    =    [   [0],   [1],   [1],   [0]]

    xor_nn = create_xor_nn(verbose=True)
    train_xor(xor_nn, X_train, y_train, epochs=1000, target_error=0.25, learning_rate=1, verbose=True)

    print("🔹 Testando Predições\n")
    results = xor_nn.predict(X_test)
    print("\n   Summary")
    for case, pred, expected in zip(X_test, results, y_test):
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
