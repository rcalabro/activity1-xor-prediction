import numpy as np

def _weighted_mean(values, weights):
    total = np.sum(weights)
    return np.sum(values * weights) / total if total > 0 else 0.0

def accuracy(matrix):
    """
    Overview: Mede a proporção de previsões corretas entre todas as previsões feitas.
    Casos bons para uso: Quando as classes estão balanceadas e você quer uma visão geral da performance.
    Quando evitar: Em problemas com classes desbalanceadas, pode mascarar desempenho ruim.
    """
    return np.trace(matrix) / np.sum(matrix)


def precision(matrix, cls=None):
    """
    Overview: Mede a proporção de previsões positivas que realmente eram positivas.
    Casos bons para uso: Quando o custo de falsos positivos é alto (ex: diagnósticos incorretos).
    Quando evitar: Quando falsos negativos são mais críticos do que falsos positivos.

    Args:
        matrix: matriz de confusão
        cls: índice da classe para retornar valor específico (opcional)

    Returns:
        precision por classe (array) ou de uma única classe (float)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.diag(matrix) / matrix.sum(axis=0)
        prec = np.nan_to_num(prec)

    if cls is not None:
        return prec[cls]

    return prec


def mean_precision(matrix, average='macro'):
    """
    Overview: Média da precisão entre classes (macro ou ponderada).
    Casos bons para uso: Para obter um score único representando a qualidade das previsões.

    Args:
        matrix: matriz de confusão
        average: 'macro' (média simples) ou 'weighted' (ponderada pelo suporte)

    Returns:
        média da precisão (float)
    """
    prec = precision(matrix)

    if average == 'macro':
        return np.mean(prec)
    elif average == 'weighted':
        support = matrix.sum(axis=1)
        return _weighted_mean(prec, support)
    else:
        raise ValueError("average deve ser 'macro' ou 'weighted'")


def recall(matrix, cls=None):
    """
    Overview: Mede a proporção de positivos reais que foram corretamente identificados.
    Casos bons para uso: Quando o custo de falsos negativos é alto (ex: fraudes, doenças).
    Quando evitar: Quando falsos positivos são mais problemáticos do que falsos negativos.

    Args:
        matrix: matriz de confusão
        cls: índice da classe para retornar valor específico (opcional)

    Returns:
        recall por classe (array) ou de uma única classe (float)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = np.diag(matrix) / matrix.sum(axis=1)
        rec = np.nan_to_num(rec)

    if cls is not None:
        return rec[cls]

    return rec


def mean_recall(matrix, average='macro'):
    """
    Overview: Média da recall entre classes (macro ou ponderada).
    Casos bons para uso: Para obter uma medida geral de cobertura do modelo sobre as classes reais.

    Args:
        matrix: matriz de confusão
        average: 'macro' (média simples) ou 'weighted' (ponderada pelo suporte)

    Returns:
        média da recall (float)
    """
    rec = recall(matrix)

    if average == 'macro':
        return np.mean(rec)
    elif average == 'weighted':
        support = matrix.sum(axis=1)
        return _weighted_mean(rec, support)
    else:
        raise ValueError("average deve ser 'macro' ou 'weighted'")