"""Executa o pipeline completo e salva as tabelas de resultados.

Gera em `results/` os mesmos números mostrados no notebook. As figuras
são exportadas do próprio notebook com `src/exportar_figuras.py`.

Uso (a partir da raiz do repositório):
    python src/train.py
"""
import warnings

import pandas as pd
from sklearn.inspection import permutation_importance

from config import PASTA_RESULTADOS, SEED, TARGET
from modeling import (
    avaliar_cv, avaliar_teste, criar_modelos, escolher_limiares, otimizar,
)
from preprocessing import carregar_dados, criar_features, separar_treino_teste

warnings.filterwarnings("ignore")


def salvar(tabela, nome):
    """Salva uma tabela em `results/` e mostra o caminho."""
    caminho = PASTA_RESULTADOS / nome
    tabela.to_csv(caminho)
    print(f"   salvo: results/{nome}")


def tabela_divisao(y, y_treino, y_teste):
    """Quantidade e % de vinhos de cada classe em cada grupo."""
    grupos = {"Base completa": y, "Treino": y_treino, "Teste": y_teste}
    linhas = {
        nome: {
            "Vinhos": len(serie),
            "Baixa/média (0)": int((serie == 0).sum()),
            "Alta (1)": int((serie == 1).sum()),
            "% alta": round(serie.mean() * 100, 1),
        }
        for nome, serie in grupos.items()
    }
    return pd.DataFrame.from_dict(linhas, orient="index")


def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("1/5 Carregando e dividindo os dados...")
    df = criar_features(carregar_dados())
    X_treino, X_teste, y_treino, y_teste = separar_treino_teste(df)
    salvar(tabela_divisao(df[TARGET], y_treino, y_teste),
           "divisao_treino_teste.csv")

    print("2/5 Comparando modelos (validação cruzada)...")
    modelos = criar_modelos()
    media_cv, rodadas_cv = avaliar_cv(modelos, X_treino, y_treino)
    salvar(media_cv, "metricas_validacao_cruzada.csv")
    salvar(rodadas_cv.set_index("Modelo"), "validacao_cruzada_rodadas.csv")

    print("3/5 Ajustando hiperparâmetros e pontos de corte...")
    otimizados, parametros, notas, combinacoes = otimizar(
        modelos, X_treino, y_treino
    )
    limiares = escolher_limiares(otimizados, X_treino, y_treino)
    ajustes = pd.DataFrame({
        "melhores_parametros": pd.Series(parametros).astype(str),
        "roc_auc_validacao": pd.Series(notas).round(3),
        "ponto_de_corte": pd.Series(limiares),
    })
    salvar(ajustes, "ajustes_modelos.csv")
    salvar(combinacoes.set_index("Modelo"), "ajuste_fino_combinacoes.csv")

    print("4/5 Avaliando no conjunto de teste...")
    com_referencia = {"Referência": modelos["Referência"], **otimizados}
    teste, _ = avaliar_teste(com_referencia,
                             {"Referência": 0.5, **limiares},
                             X_treino, y_treino, X_teste, y_teste)
    salvar(teste, "metricas_teste.csv")

    print("5/5 Medindo a influência de cada variável...")
    perm = permutation_importance(otimizados["Random Forest"], X_teste,
                                  y_teste, scoring="roc_auc", n_repeats=30,
                                  random_state=SEED)
    importancia = pd.Series(perm.importances_mean, index=X_teste.columns)
    importancia = importancia.sort_values(ascending=False).round(4)
    salvar(importancia.to_frame("queda_roc_auc"),
           "importancia_variaveis.csv")

    print("\nMétricas no teste:\n", teste)
    print("\nPara exportar as figuras do notebook: "
          "python src/exportar_figuras.py")


if __name__ == "__main__":
    main()
