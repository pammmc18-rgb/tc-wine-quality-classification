"""Configurações gerais do projeto: caminhos, semente e paleta de cores."""
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ / "data" / "WineQT.csv"
PASTA_RESULTADOS = RAIZ / "results"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"

SEED = 42
TARGET = "alta_qualidade"
NOTA_CORTE = 7

# Paleta inspirada em vinho
BORDO = "#6D071A"          # vinho tinto – cor principal
TINTO = "#9E2A3A"          # tinto mais claro
VERDE_GARRAFA = "#2F4F3E"  # garrafa
DOURADO = "#C9A227"        # vinho branco
CORTICA = "#C8B49A"        # rolha – tom neutro
ROSE_CLARO = "#F6E3E6"     # rosé bem claro
TEXTO = "#2B1A1D"          # quase preto, puxado para o vinho

CORES = [CORTICA, BORDO]  # 0 = baixa/média | 1 = alta
CORES_MODELOS = {
    "Regressão Logística": DOURADO,
    "SVM": VERDE_GARRAFA,
    "Random Forest": BORDO,
}
CMAP_DIV = LinearSegmentedColormap.from_list(
    "vinho_div", [VERDE_GARRAFA, "#ffffff", BORDO]
)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "vinho_seq", ["#ffffff", ROSE_CLARO, BORDO]
)
TEMA = {
    "text.color": TEXTO,
    "axes.labelcolor": TEXTO,
    "axes.titlecolor": TEXTO,
    "axes.edgecolor": CORTICA,
    "xtick.color": TEXTO,
    "ytick.color": TEXTO,
}
