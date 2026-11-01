"""Carga, limpeza, feature engineering e pré-processamento dos dados."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import CAMINHO_DADOS, NOTA_CORTE, SEED, TARGET

NOVAS = ["so2_livre_pct", "alcool_x_sulfatos", "alcool_por_volatil"]
ASSIMETRICAS = [
    "residual sugar", "chlorides", "sulphates", "free sulfur dioxide",
    "total sulfur dioxide", "alcool_por_volatil",
]


def carregar_dados(caminho=CAMINHO_DADOS):
    """Lê a base, remove `Id` e duplicatas e cria a variável alvo binária."""
    df = pd.read_csv(caminho).drop(columns=["Id"])
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df["quality"] >= NOTA_CORTE).astype(int)
    return df


def criar_features(df):
    """Adiciona as três features combinadas usadas nos modelos."""
    df = df.copy()
    df["so2_livre_pct"] = (df["free sulfur dioxide"]
                           / df["total sulfur dioxide"])
    df["alcool_x_sulfatos"] = df["alcohol"] * df["sulphates"]
    df["alcool_por_volatil"] = df["alcohol"] / df["volatile acidity"]
    return df


def separar_treino_teste(df, test_size=0.2):
    """Divide em treino e teste de forma estratificada."""
    X = df.drop(columns=["quality", TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=SEED)


def criar_preprocessador():
    """Log nas variáveis assimétricas + padronização (média 0, desvio 1)."""
    log = FunctionTransformer(np.log, feature_names_out="one-to-one")
    return make_pipeline(
        ColumnTransformer(
            [("log", log, ASSIMETRICAS)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
        StandardScaler(),
    )
