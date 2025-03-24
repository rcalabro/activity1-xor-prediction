import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcula a matriz de confusão para classificação binária ou multiclasse.

    Args:
        y_true: Lista ou array de rótulos reais.
        y_pred: Lista ou array de rótulos previstos.
        labels: Lista de rótulos possíveis. Se None, será inferida automaticamente.

    Returns:
        matrix: Matriz de confusão.
        labels: Lista de rótulos usados como referência (na ordem das linhas e colunas).
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    assert len(y_true) == len(y_pred), "y_true e y_pred devem ter o mesmo comprimento"

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    labels_index = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        i = labels_index[t]
        j = labels_index[p]
        matrix[i, j] += 1

    return matrix, labels