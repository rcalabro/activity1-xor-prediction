import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_network(nn,
                 X,
                 width=800,
                 height=600,
                 max_show_input=8,
                 max_show_hidden=8,
                 max_show_output=None,
                 title=""):
    """
    Desenha a rede neural 'nn', exibindo até um certo número de neurônios
    em cada camada (com truncamento) e colorindo cada neurônio em escala de cinza
    conforme a ativação. Ajusta manualmente as margens (subplots_adjust) e
    aumenta o espaçamento interno entre neurônios/camadas.

    Parâmetros:
      - nn: instância de NeuralNetwork (com nn.weights e nn.biases).
      - X: array NumPy de shape (1, input_dim). Se vier mais de 1 amostra, usa só a primeira.
      - width, height (int): tamanho da figura em pixels (default maior para facilitar).
      - max_show_input (int ou None): max. neurônios na camada de entrada (None => todos).
      - max_show_hidden (int ou None): max. neurônios nas camadas ocultas (None => todos).
      - max_show_output (int ou None): max. neurônios na camada de saída (None => todos).
    """

    # Se chegar mais de uma amostra, usar só a primeira
    if X.shape[0] > 1:
        X = X[:1]

    # ========== Ajustes de espaçamento e estilo ==========
    # Aumente ou diminua conforme necessidade
    x_spacing = 12.0    # distância horizontal entre camadas (BEM maior que antes)
    y_spacing = 8.0    # distância vertical entre neurônios
    neuron_size = 500 # círculos maiores

    # Converte width/height em polegadas, dpi=100
    fig_width_inch = width / 100
    fig_height_inch = height / 100

    # Cria a figura e eixos (sem constrained_layout nem tight_layout)
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=100)

    # Ajusta manualmente as margens do subplot para “apertar” o branco em volta
    # left, right, top, bottom => frações do tamanho total da figura
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)

    # ========== 1) Forward pass para obter ativações ==========
    activations_per_layer = [X]
    A = X
    for w, b in zip(nn.weights, nn.biases):
        Z = np.dot(A, w) + b
        A = nn.activation_func(Z)
        activations_per_layer.append(A)

    layer_sizes = [nn.input_layer] + nn.hidden_layers + [nn.output_layer]
    total_layers = len(layer_sizes)

    # ========== 2) Definir quantos neurônios mostrar em cada camada ==========

    def get_max_show_for_layer(layer_idx):
        """
        Retorna o limite de neurônios para a camada layer_idx:
          - 0 => camada de entrada => max_show_input
          - total_layers-1 => camada de saída => max_show_output
          - senão => camada oculta => max_show_hidden
        """
        if layer_idx == 0:
            return max_show_input
        elif layer_idx == total_layers - 1:
            return max_show_output
        else:
            return max_show_hidden

    def truncated_indices(size, max_show):
        """
        Retorna a lista de índices a exibir e se houve truncamento.
        Se max_show=None ou size <= max_show => [0..size-1].
        Caso contrário => primeiros e últimos (max_show//2),
        com um neurônio 'falso'=-1 no meio para '...'.
        """
        if max_show is None or size <= max_show:
            return list(range(size)), False
        half = max_show // 2
        idxs = list(range(half)) + [-1] + list(range(size - half, size))
        return idxs, True

    # ========== 3) Normalizar pesos/bias (para cor/espessura) e descobrir faixa de ativação ==========
    all_weights = np.concatenate([w.flatten() for w in nn.weights])
    w_min, w_max = all_weights.min(), all_weights.max()
    if w_min == w_max:
        w_min, w_max = w_min - 1e-9, w_max + 1e-9

    all_biases = np.concatenate([b.flatten() for b in nn.biases])
    b_min, b_max = all_biases.min(), all_biases.max()
    if b_min == b_max:
        b_min, b_max = b_min - 1e-9, b_max + 1e-9

    # Faixa de ativação (para pintar neurônios em cinza)
    all_activations = []
    for act in activations_per_layer:
        all_activations.extend(act[0].tolist())
    all_activations = np.array(all_activations)
    A_min, A_max = all_activations.min(), all_activations.max()
    if A_min == A_max:
        A_min, A_max = A_min - 1e-9, A_max + 1e-9

    ax.set_title("NeuralNetwork: " + title + "\n" + "weigth: cor | bias: espessura")

    neuron_positions = []

    # Colormap para ativação em escala de cinza
    gray_cmap = cm.get_cmap('Greys_r')

    # ========== 4) Plotar neurônios nas camadas ==========
    for layer_idx, size in enumerate(layer_sizes):
        max_show = get_max_show_for_layer(layer_idx)
        idxs, _ = truncated_indices(size, max_show)

        y_start = -(len(idxs) - 1) * y_spacing / 2.0
        layer_positions = {}

        layer_acts = activations_per_layer[layer_idx][0]

        for j, idx in enumerate(idxs):
            x_coord = layer_idx * x_spacing
            y_coord = y_start + j * y_spacing

            if idx == -1:
                # Neurônio "falso" => reticências
                ax.scatter(
                    x_coord, y_coord,
                    s=neuron_size * 0.7,
                    facecolors='lightgray',
                    edgecolors='black'
                )
                ax.text(x_coord, y_coord, "...", ha='center', va='center')
                layer_positions[idx] = (x_coord, y_coord)

            else:
                act_val = layer_acts[idx]
                norm_act = (act_val - A_min) / (A_max - A_min)
                bg_color = gray_cmap(norm_act)
                text_color = 'white' if norm_act < 0.5 else 'black'

                ax.scatter(
                    x_coord, y_coord,
                    s=neuron_size,
                    facecolors=bg_color,
                    edgecolors='black',
                    zorder=2
                )
                ax.text(
                    x_coord, y_coord,
                    f"{act_val:.2f}",
                    ha='center', va='center',
                    color=text_color,
                    zorder=3
                )

                layer_positions[idx] = (x_coord, y_coord)

        neuron_positions.append(layer_positions)

    # ========== 5) Desenhar conexões (setas) ==========
    cmap_arrows = cm.get_cmap('RdYlGn')
    lw_min, lw_max = 0.5, 4.0

    for layer_i in range(total_layers - 1):
        w_matrix = nn.weights[layer_i]
        b_vector = nn.biases[layer_i]

        src_positions = neuron_positions[layer_i]
        dst_positions = neuron_positions[layer_i + 1]

        src_max = get_max_show_for_layer(layer_i)
        dst_max = get_max_show_for_layer(layer_i + 1)
        src_idxs, _ = truncated_indices(layer_sizes[layer_i], src_max)
        dst_idxs, _ = truncated_indices(layer_sizes[layer_i+1], dst_max)

        for s_idx in src_idxs:
            if s_idx == -1:
                continue
            x1, y1 = src_positions[s_idx]

            for d_idx in dst_idxs:
                if d_idx == -1:
                    continue
                x2, y2 = dst_positions[d_idx]

                w_val = w_matrix[s_idx, d_idx]
                norm_w = (w_val - w_min) / (w_max - w_min)
                arrow_color = cmap_arrows(norm_w)

                b_val = b_vector[0, d_idx]
                norm_b = (b_val - b_min) / (b_max - b_min)
                lw = lw_min + norm_b*(lw_max - lw_min)

                dx, dy = x2 - x1, y2 - y1
                ax.arrow(
                    x1, y1, dx, dy,
                    head_width=0.2,
                    length_includes_head=True,
                    linewidth=lw,
                    color=arrow_color,
                    alpha=0.9,
                    zorder=1
                )

    ax.set_xlabel("Camada")
    ax.set_ylabel("Neurônios (truncados)")

    # ========== 6) Barra de cores (pesos) ==========
    norm_w = mcolors.Normalize(vmin=w_min, vmax=w_max)
    sm = cm.ScalarMappable(norm=norm_w, cmap=cmap_arrows)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Peso (vermelho=menor, verde=maior)")

    # Ajuste manual dos limites do plot
    ax.set_xlim(-x_spacing, x_spacing*(total_layers - 1) + x_spacing)

    def max_layer_plotted():
        """
        Retorna quantos neurônios, no máximo, estão sendo plotados em alguma camada
        (após truncamento), para dimensionar verticalmente.
        """
        max_count = 0
        for i, size in enumerate(layer_sizes):
            ms = get_max_show_for_layer(i)
            idxs, _ = truncated_indices(size, ms)
            if len(idxs) > max_count:
                max_count = len(idxs)
        return max_count

    mlp = max_layer_plotted()
    ax.set_ylim(-mlp*y_spacing, mlp*y_spacing)

    plt.show()
