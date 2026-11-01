"""Definição, ajuste e avaliação dos modelos de classificação."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from config import SEED
from preprocessing import criar_preprocessador

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
METRICAS = {
    "Acurácia": "accuracy",
    "Precisão": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC-AUC": "roc_auc",
}
GRIDS = {
    "Regressão Logística": {"logisticregression__C": [0.001, 0.01, 0.1, 1]},
    "SVM": {
        "svc__C": [0.01, 0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1],
    },
    "Random Forest": {
        "max_depth": [None, 8, 12],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.5],
    },
}


def criar_modelos():
    """Retorna a referência (Dummy) e os três modelos comparados."""
    return {
        "Referência": DummyClassifier(strategy="most_frequent"),
        "Regressão Logística": make_pipeline(
            criar_preprocessador(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "SVM": make_pipeline(
            criar_preprocessador(),
            SVC(kernel="rbf", class_weight="balanced", probability=True,
                random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED,
            n_jobs=-1,
        ),
    }


def avaliar_cv(modelos, X, y):
    """Médias por modelo e ROC-AUC de cada rodada da validação cruzada."""
    tabela, rodadas = {}, []
    for nome, modelo in modelos.items():
        res = cross_validate(modelo, X, y, cv=CV, scoring=METRICAS)
        tabela[nome] = {k: res[f"test_{k}"].mean() for k in METRICAS}
        for i, nota in enumerate(res["test_ROC-AUC"], start=1):
            rodadas.append({"Modelo": nome, "Rodada": i, "ROC-AUC": nota})
    return pd.DataFrame(tabela).T.round(3), pd.DataFrame(rodadas).round(4)


def otimizar(modelos, X, y):
    """Busca em grade pela maior ROC-AUC.

    Retorna os modelos ajustados, os melhores parâmetros, a melhor ROC-AUC
    de cada modelo e a tabela com todas as combinações testadas.
    """
    otimizados, parametros, notas, combinacoes = {}, {}, {}, []
    for nome, grid in GRIDS.items():
        busca = GridSearchCV(modelos[nome], grid, scoring="roc_auc", cv=CV,
                             n_jobs=-1)
        busca.fit(X, y)
        otimizados[nome] = busca.best_estimator_
        parametros[nome] = busca.best_params_
        notas[nome] = busca.best_score_
        res = pd.DataFrame(busca.cv_results_)
        combinacoes.append(pd.DataFrame({
            "Modelo": nome,
            "Parâmetros": res["params"].astype(str),
            "ROC-AUC média": res["mean_test_score"].round(4),
            "Posição": res["rank_test_score"],
        }))
    return otimizados, parametros, notas, pd.concat(combinacoes)


def escolher_limiares(modelos, X, y):
    """Ponto de corte que maximiza o F1 em validação cruzada no treino."""
    grade = np.arange(0.10, 0.90, 0.01)
    limiares = {}
    for nome, modelo in modelos.items():
        proba = cross_val_predict(modelo, X, y, cv=CV,
                                  method="predict_proba")[:, 1]
        f1s = [f1_score(y, proba >= t) for t in grade]
        limiares[nome] = round(float(grade[np.argmax(f1s)]), 2)
    return limiares


def avaliar_teste(modelos, limiares, X_treino, y_treino, X_teste, y_teste):
    """Treina no treino e mede no teste; retorna métricas e probabilidades."""
    linhas, probas = {}, {}
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        proba = modelo.predict_proba(X_teste)[:, 1]
        pred = proba >= limiares[nome]
        probas[nome] = proba
        linhas[nome] = {
            "Acurácia": accuracy_score(y_teste, pred),
            "Precisão": precision_score(y_teste, pred, zero_division=0),
            "Recall": recall_score(y_teste, pred),
            "F1": f1_score(y_teste, pred),
            "ROC-AUC": roc_auc_score(y_teste, proba),
        }
    tabela = pd.DataFrame(linhas).T.round(3)
    return tabela.sort_values("ROC-AUC", ascending=False), probas
