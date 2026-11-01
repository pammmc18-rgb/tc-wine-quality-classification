"""Exporta todas as figuras do notebook executado para `results/figuras/`.
 
As figuras saem do próprio notebook, então `results/` mostra exatamente
os mesmos gráficos da análise. Rode depois de executar e salvar o notebook.
Precisa apenas da biblioteca `nbformat` (pip install nbformat).
 
Uso (a partir da raiz do repositório):
    python src/exportar_figuras.py
"""
import base64
from pathlib import Path
 
import nbformat
 
RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_NOTEBOOK = RAIZ / "notebooks" / "WineQT.ipynb"
PASTA_FIGURAS = RAIZ / "results" / "figuras"
 
# Nome de cada figura, na ordem em que aparece no notebook
NOMES = [
    "01_balanceamento_classes",
    "02_distribuicao_variaveis",
    "03_medidas_por_classe",
    "04_correlacao",
    "05_outliers_percentual",
    "06_outliers_por_caracteristica",
    "07_limpeza_duplicatas",
    "08_novas_medidas",
    "09_divisao_treino_teste",
    "10_notas_treino_teste",
    "11_medidas_treino_teste",
    "12_padronizacao_antes_depois",
    "13_assimetria_log",
    "14_esquema_validacao_cruzada",
    "15_comparacao_modelos_cv",
    "16_efeito_novas_medidas",
    "17_ajuste_fino",
    "18_ponto_de_corte",
    "19_matriz_confusao",
    "20_curva_roc_metricas",
    "21_validacao_x_teste",
    "22_importancia_variaveis",
    "23_efeito_parcial",
]
 
 
def extrair_imagens(caminho):
    """Lista as imagens PNG das saídas do notebook, na ordem."""
    notebook = nbformat.read(caminho, as_version=4)
    imagens = []
    for celula in notebook.cells:
        if celula.cell_type != "code":
            continue
        for saida in celula.get("outputs", []):
            png = saida.get("data", {}).get("image/png")
            if png:
                imagens.append(base64.b64decode(png))
    return imagens
 
 
def main():
    print(f"Lendo: {CAMINHO_NOTEBOOK}")
    imagens = extrair_imagens(CAMINHO_NOTEBOOK)
    if not imagens:
        raise SystemExit("O notebook não tem figuras: execute-o e salve.")
    if len(imagens) != len(NOMES):
        print(f"Aviso: o notebook tem {len(imagens)} figuras e a lista "
              f"de nomes tem {len(NOMES)}. Usando numeração simples.")
        nomes = [f"{i:02d}_figura" for i in range(1, len(imagens) + 1)]
    else:
        nomes = NOMES
 
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    for antiga in PASTA_FIGURAS.glob("*.png"):
        antiga.unlink()
    for nome, imagem in zip(nomes, imagens):
        (PASTA_FIGURAS / f"{nome}.png").write_bytes(imagem)
    print(f"{len(imagens)} figuras salvas em {PASTA_FIGURAS}")
 
 
if __name__ == "__main__":
    main()