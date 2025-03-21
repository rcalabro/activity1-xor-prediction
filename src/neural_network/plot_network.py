import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors  # Para tooltips interativos

"""
Código gerado inicialmente pelo ChatGPT e posteriormente modificado manualmente para ajustes específicos.

- Implementação original da visualização de redes neurais com matplotlib.
- Adição de tooltips interativos para exibição de pesos e bias ao passar o mouse.
- Ajustes no layout para otimização do espaço e melhor organização visual.
- Alterações na lógica de exibição para garantir que os tooltips apareçam apenas ao passar sobre as conexões.

Desenvolvido com base nos requisitos do usuário, mantendo um código claro, modular e interativo.

Autor: Renato Calabro
"""


def plot_network(nn,
                 x_input,
                 width=1400,
                 height=800,
                 max_show_input=8,
                 max_show_hidden=8,
                 max_show_output=None,
                 show_tooltip=True,
                 title="Neural Network",
                 show=True):
    """
    Desenha a rede neural 'nn', exibindo até um certo número de neurônios
    em cada camada (com truncamento). A opacidade das conexões representa o bias,
    e a cor representa o peso. Tooltips interativos mostram peso e bias ao passar o mouse.

    Parâmetros:
      - nn: instância de NeuralNetwork (com nn.weights e nn.biases).
      - x_input: array NumPy de shape (1, input_dim). Se vier mais de 1 amostra, usa só a primeira.
      - width, height (int): tamanho da figura em pixels.
      - max_show_input (int ou None): máx. neurônios na camada de entrada (None => todos).
      - max_show_hidden (int ou None): máx. neurônios nas camadas ocultas (None => todos).
      - max_show_output (int ou None): máx. neurônios na camada de saída (None => todos).
      - title (str): título principal do gráfico (default: "Neural Network").
    """

    if x_input.shape[0] > 1:
        x_input = x_input[:1]

    # ========== Ajustes de espaçamento e estilo ==========
    x_spacing = 12.0   # Distância horizontal entre camadas
    y_spacing = 8.0    # Distância vertical entre neurônios
    neuron_size = 500  # Tamanho dos neurônios
    arrow_width = 2.0  # Espessura fixa das setas

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)

    # ========== Forward pass ==========
    activations_per_layer = [x_input]
    a = x_input
    for w, b in zip(nn.weights, nn.biases):
        z = np.dot(a, w) + b
        a = nn.activation_func(z)
        activations_per_layer.append(a)

    layer_sizes = [nn.input_layer] + nn.hidden_layers + [nn.output_layer]
    total_layers = len(layer_sizes)

    def get_max_show_for_layer(layer_idx):
        if layer_idx == 0:
            return max_show_input
        elif layer_idx == total_layers - 1:
            return max_show_output
        return max_show_hidden

    def truncated_indices(size, max_show):
        if max_show is None or size <= max_show:
            return list(range(size)), False
        half = max_show // 2
        idxs = list(range(half)) + [-1] + list(range(size - half, size))
        return idxs, True

    # ========== Normalização ==========
    all_weights = np.concatenate([w.flatten() for w in nn.weights])
    w_min, w_max = all_weights.min(), all_weights.max()
    all_biases = np.concatenate([b.flatten() for b in nn.biases])
    b_min, b_max = all_biases.min(), all_biases.max()
    all_activations = np.concatenate([a.flatten() for a in activations_per_layer])
    a_min, a_max = all_activations.min(), all_activations.max()

    ax.set_title(f"{title}\nweight: cor | bias: opacidade")

    neuron_positions = []
    gray_cmap = cm.get_cmap('Greys_r')

    # ========== Plotar neurônios ==========
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
                ax.scatter(x_coord, y_coord, s=neuron_size * 0.7, facecolors='lightgray', edgecolors='black')
                ax.text(x_coord, y_coord, "...", ha='center', va='center')
                layer_positions[idx] = (x_coord, y_coord)
            else:
                act_val = layer_acts[idx]
                norm_act = (act_val - a_min) / (a_max - a_min)
                bg_color = gray_cmap(norm_act)
                text_color = 'white' if norm_act < 0.5 else 'black'

                ax.scatter(x_coord, y_coord, s=neuron_size, facecolors=bg_color, edgecolors='black', zorder=2)
                ax.text(x_coord, y_coord, f"{act_val:.2f}", ha='center', va='center', color=text_color, zorder=3)
                layer_positions[idx] = (x_coord, y_coord)

        neuron_positions.append(layer_positions)

    # ========== Plotar conexões ==========
    cmap_arrows = cm.get_cmap('RdYlGn')
    arrows = []
    epsilon = 1e-9  # 🔹 Pequeno deslocamento para evitar divisão por zero

    for layer_i in range(total_layers - 1):
        w_matrix = nn.weights[layer_i]
        b_vector = nn.biases[layer_i]

        src_positions = neuron_positions[layer_i]
        dst_positions = neuron_positions[layer_i + 1]

        for s_idx, (x1, y1) in src_positions.items():
            if s_idx == -1:
                continue

            for d_idx, (x2, y2) in dst_positions.items():
                if d_idx == -1:
                    continue

                w_val = w_matrix[s_idx, d_idx]
                norm_w = (w_val - w_min) / max(epsilon, w_max - w_min)
                arrow_color = cmap_arrows(norm_w)

                b_val = b_vector[0, d_idx]
                norm_b = (b_val - b_min) / max(epsilon, b_max - b_min)
                arrow_alpha = 0.2 + 0.8 * norm_b

                arrow = ax.arrow(
                    x1, y1, x2-x1, y2-y1,
                    head_width=0.2,
                    linewidth=arrow_width,
                    color=arrow_color,
                    alpha=arrow_alpha,
                    zorder=1
                )

                if show_tooltip:
                    # Salva a seta e os valores para o tooltip
                    arrows.append((arrow, w_val, b_val))

    # ========== Adiciona tooltip interativo ==========
    cursor = mplcursors.cursor([a[0] for a in arrows], hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        idx = [a[0] for a in arrows].index(sel.artist)
        w_val, b_val = arrows[idx][1], arrows[idx][2]
        sel.annotation.set_text(f"w: {w_val:.4f} | B: {b_val:.4f}")
        sel.annotation.set_visible(True)  # Garante que só aparece quando hover

    @cursor.connect("remove")
    def on_leave(sel):
        sel.annotation.set_visible(False)  # Esconde o tooltip ao sair

    # Adiciona legenda da escala de cores dos pesos
    sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=w_min, vmax=w_max), cmap=cmap_arrows)
    fig.colorbar(sm, ax=ax, label="Peso (vermelho=menor, verde=maior)")

    if show:
        plt.show()
