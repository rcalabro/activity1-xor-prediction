import matplotlib.pyplot as plt
import os
import re
import unicodedata

def normalize_text(text):
    """
    Converte texto para lowercase_snake_case sem acentos ou caracteres especiais.
    """
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text


def export_plots(output_dir, use_title=False, prefix="fig"):
    """
    Salva todas as figuras abertas do Matplotlib em arquivos PNG no diretório especificado.

    Args:
        output_dir (str): Caminho do diretório onde os PNGs serão salvos.
        prefix (str): Prefixo do nome do arquivo (default: "fig").
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        ax = fig.axes[0] if fig.axes else None

        title = ax.get_title() if ax and use_title else f"{i+1}"
        safe_title = normalize_text(title) or "no_title"

        filename = f"{prefix}_{safe_title}.png"
        path = os.path.join(output_dir, filename)

        fig.savefig(path)
        print(f"Salvo: {path}")

    plt.close("all")
