import numpy as np

def accuracy(matrix):
    return np.trace(matrix) / np.sum(matrix)