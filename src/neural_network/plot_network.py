import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_network(nn, X, width=800, height=600):
    """
    Desenha a rede neural 'nn' como um grafo, mostrando cada neurônio em cada camada.
    Características:
      - Conexões são setas coloridas em vermelho->verde conforme o peso (RdYlGn).
      - Espessura (linewidth) das setas depende do bias do neurônio de destino.
      - Cada neurônio é um círculo maior, com a ativação servindo de cor de fundo (escala de cinza):
          Quanto maior a ativação, mais para branco;
          Quanto menor, mais para preto.
      - A cor do texto (valor de ativação) é branca se o fundo estiver escuro, e preta se estiver claro.
      - Largura e altura da figura em pixels (default 800x600).
      - Barra de cores à direita indicando o range de pesos.

    Parâmetros:
      - nn: instância de NeuralNetwork (com nn.weights e nn.biases).
      - X: array NumPy de shape (1, input_dim) para 1 amostra (se vier mais, usa só a primeira).
      - width (int): Largura da figura em pixels.
      - height (int): Altura da figura em pixels.
    """

    # Se chegar mais de uma amostra, usar apenas a primeira
    if X.shape[0] > 1:
        X = X[:1]

    # ====== Ajustes de espaçamento ======
    x_spacing = 4.0
    y_spacing = 3.0
    neuron_size = 1000  # tamanho do círculo do neurônio

    # Converte width/height para polegadas e define dpi=100
    fig_width_inch = width / 100
    fig_height_inch = height / 100

    # Cria figura e eixos explicitamente
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=100)

    # 1) Forward manual para obter as ativações em cada camada
    activations_per_layer = [X]
    A = X
    for w, b in zip(nn.weights, nn.biases):
        Z = np.dot(A, w) + b
        A = nn.activation_func(Z)
        activations_per_layer.append(A)

    # 2) Descobrir quantos neurônios em cada camada
    layer_sizes = [nn.inputLayer] + nn.hiddenLayers + [nn.outputLayer]
    total_layers = len(layer_sizes)

    # 3) Juntar todos os pesos e biases para normalizar (para cor e espessura das setas)
    all_weights = np.concatenate([w.flatten() for w in nn.weights])
    w_min, w_max = all_weights.min(), all_weights.max()
    if w_min == w_max:
        w_min, w_max = w_min - 1e-9, w_max + 1e-9

    all_biases = np.concatenate([b.flatten() for b in nn.biases])
    b_min, b_max = all_biases.min(), all_biases.max()
    if b_min == b_max:
        b_min, b_max = b_min - 1e-9, b_max + 1e-9

    ax.set_title("Visualização da Rede Neural (Peso: setas verm->verd / Bias: espessura / Ativação: cinza)")

    # 4) Vamos também juntar todas as ativações para normalizar cor de fundo do neurônio
    #    (assim, definimos qual é a menor e maior ativação na rede)
    all_activations = []
    for act in activations_per_layer:
        # Cada 'act' tem shape (1, layer_sizes[i]), pegamos o [0] para virar 1D
        all_activations.extend(act[0].tolist())
    all_activations = np.array(all_activations)
    A_min, A_max = all_activations.min(), all_activations.max()
    if A_min == A_max:
        A_min, A_max = A_min - 1e-9, A_max + 1e-9

    neuron_positions = []

    # Cria um colormap para os neurônios (preto->branco)
    # 'Greys_r' faz: 0 => preto e 1 => branco.
    gray_cmap = cm.get_cmap('Greys_r')

    # 5) Desenhar neurônios (círculos) e anotar ativação dentro
    for i, size in enumerate(layer_sizes):
        y_start = -(size - 1) * y_spacing / 2.0
        layer_positions = []
        for n_idx in range(size):
            x_coord = i * x_spacing
            y_coord = y_start + n_idx * y_spacing

            # Valor de ativação
            act_val = activations_per_layer[i][0, n_idx]
            # Normaliza para [0,1] => cor de fundo em cinza
            norm_act = (act_val - A_min) / (A_max - A_min)
            bg_color = gray_cmap(norm_act)

            # Escolha da cor da fonte: se norm_act < 0.5 (fundo escuro), fonte branca; senão, preta
            text_color = 'white' if norm_act < 0.5 else 'black'

            # Desenha o neurônio como um círculo maior, definindo a cor de fundo
            ax.scatter(
                x_coord,
                y_coord,
                s=neuron_size,
                facecolors=bg_color,   # Cor do interior do círculo
                edgecolors='black',
                zorder=2
            )

            # Texto da ativação (no centro do círculo)
            ax.text(
                x_coord, y_coord,
                f"{act_val:.2f}",
                ha='center', va='center',
                color=text_color,
                zorder=3
            )

            layer_positions.append((x_coord, y_coord))
        neuron_positions.append(layer_positions)

    # 6) Desenhar conexões (setas) usando colormap RdYlGn para os pesos
    cmap = cm.get_cmap('RdYlGn')
    lw_min, lw_max = 0.5, 5.0

    for i in range(total_layers - 1):
        w_matrix = nn.weights[i]
        b_vector = nn.biases[i]  # shape (1, layer_sizes[i+1])
        for n_idx, (x1, y1) in enumerate(neuron_positions[i]):
            for m_idx, (x2, y2) in enumerate(neuron_positions[i + 1]):
                w_value = w_matrix[n_idx, m_idx]
                # Normaliza peso [w_min, w_max] -> [0,1] para cor
                norm_w = (w_value - w_min) / (w_max - w_min)
                arrow_color = cmap(norm_w)

                # Bias do neurônio de destino
                b_value = b_vector[0, m_idx]
                norm_b = (b_value - b_min) / (b_max - b_min)
                lw = lw_min + norm_b * (lw_max - lw_min)

                dx = x2 - x1
                dy = y2 - y1

                ax.arrow(
                    x1, y1, dx, dy,
                    head_width=0.15,
                    length_includes_head=True,
                    linewidth=lw,
                    color=arrow_color,
                    alpha=0.9,
                    zorder=1
                )

    ax.set_xlabel("Camada")
    ax.set_ylabel("Posição do neurônio (eixo Y)")

    # 7) Barra de cores (legenda) para os pesos (setas)
    norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Valor do peso (vermelho: menor, verde: maior)")

    # Ajuste de limites para espaçamento extra
    ax.set_xlim(-x_spacing, x_spacing*(total_layers - 1) + x_spacing)
    max_neurons = max(layer_sizes)
    ax.set_ylim(-max_neurons*y_spacing, max_neurons*y_spacing)

    plt.show()
