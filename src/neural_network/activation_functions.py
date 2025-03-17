import numpy as np
from scipy.special import expit

"""
Este módulo foi gerado inicialmente pelo ChatGPT e posteriormente modificado manualmente 
para aprimoramento, inclusão de mais funções e ajustes específicos.

Módulo com diversas funções de ativação usadas em redes neurais.

Inclui funções populares (ReLU, Sigmoid, Tanh, Softmax) e algumas menos comuns (Swish, ELU, Leaky ReLU, GELU).

Cada função possui:
- Overview: Explicação breve do que a função faz.
- Casos bons para uso: Quando a função é recomendada.
- Quando evitar: Situações onde não é a melhor escolha.

Autor: Renato Calabro
"""

def step_function(x):
    """
    Overview: Função degrau binária, retorna 1 para x >= 0 e 0 para x < 0.
    Casos bons para uso: Perceptrons e redes neurais simples para classificação binária.
    Quando evitar: Em redes profundas, pois não é diferenciável e não permite aprendizado eficiente.
    """
    return np.where(x > 0, 1, 0)

def step_function_derivative(x):
    """
    Overview: Derivada da função degrau, que é zero em todos os pontos (não útil para aprendizado).
    Casos bons para uso: Apenas para estudo teórico, pois não é útil no treinamento de redes neurais.
    Quando evitar: Sempre que precisar de gradientes, pois a derivada é sempre zero.
    """
    return np.zeros_like(x)

def sigmoid(x):
    """
    Overview: Função de ativação sigmoide, que mapeia qualquer valor real para o intervalo (0,1).
    Casos bons para uso: Classificação binária e redes neurais simples.
    Quando evitar: Em redes profundas, pois sofre com o problema do gradiente desaparecendo.
    """
    return expit(x)  # Evita overflow numérico com scipy.special.expit

def sigmoid_derivative(x):
    """Derivada da função Sigmoid."""
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    """
    Overview: ReLU (Rectified Linear Unit) retorna 0 para valores negativos e x para positivos.
    Casos bons para uso: Redes profundas devido à eficiência computacional e redução do gradiente desaparecendo.
    Quando evitar: Pode sofrer com o problema de neurônios mortos (valores negativos sempre retornam 0).
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivada da função ReLU."""
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    """
    Overview: Variante da ReLU que permite pequenos valores negativos em vez de 0.
    Casos bons para uso: Quando há risco de neurônios mortos em redes profundas.
    Quando evitar: Se os valores negativos não são um problema, a ReLU padrão pode ser suficiente.
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivada da função Leaky ReLU."""
    return np.where(x > 0, 1, alpha)

def tanh(x):
    """
    Overview: Mapeia valores reais para o intervalo (-1,1), sendo semelhante ao Sigmoid, mas centrado em zero.
    Casos bons para uso: Quando os dados são distribuídos em torno de 0.
    Quando evitar: Pode sofrer com o gradiente desaparecendo em redes profundas.
    """
    return np.tanh(x)

def tanh_derivative(x):
    """Derivada da função Tangente Hiperbólica."""
    return 1 - np.tanh(x)**2

def softmax(x):
    """
    Overview: Converte um vetor de valores reais em probabilidades que somam 1.
    Casos bons para uso: Classificação multiclasse na camada de saída.
    Quando evitar: Se os valores de entrada forem muito grandes, pode haver problemas de estabilidade numérica.
    """
    exps = np.exp(x - np.max(x))  # Subtrai o máximo para evitar overflow
    return exps / np.sum(exps, axis=-1, keepdims=True)

def softmax_derivative(x):
    """Derivada da função Softmax (matriz Jacobiana)."""
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

def swish(x):
    """
    Overview: Ativação não linear onde f(x) = x * sigmoid(x), permitindo valores negativos atenuados.
    Casos bons para uso: Redes profundas e modelos avançados como EfficientNet.
    Quando evitar: Pode ser computacionalmente mais caro do que ReLU.
    """
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivada da função Swish."""
    return sigmoid(x) + x * sigmoid_derivative(x)

def elu(x, alpha=1.0):
    """
    Overview: Exponential Linear Unit (ELU) resolve o problema de neurônios mortos da ReLU.
    Casos bons para uso: Redes profundas quando há muitos valores negativos.
    Quando evitar: É mais caro computacionalmente do que ReLU ou Leaky ReLU.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivada da função ELU."""
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

def gelu(x):
    """
    Overview: Gaussian Error Linear Unit (GELU), ativação suave baseada na distribuição Gaussiana.
    Casos bons para uso: Modelos de NLP, como BERT, devido à suavidade da ativação.
    Quando evitar: Em redes pequenas, onde ReLU pode ser mais eficiente.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """Derivada da função GELU."""
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))) + \
        0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))**2) * \
        (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2))

# 🔹 Dicionário para importar funções de forma dinâmica
ACTIVATIONS = {
    "step": step_function,
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softmax": softmax,
    "swish": swish,
    "elu": elu,
    "gelu": gelu
}

# 🔹 Dicionário das derivadas das
# funções de ativação (para uso futuro no treinamento)
ACTIVATION_DERIVATIVES = {
    "step": step_function_derivative,
    "sigmoid": sigmoid_derivative,
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
    "swish": swish_derivative,
    "elu": elu_derivative,
    "gelu": gelu_derivative
}
