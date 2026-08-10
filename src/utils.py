"""Funciones compartidas: semillas, métricas y carga robusta de CSVs."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from config import ID_COL, SEED, TARGET


def set_seed(seed: int = SEED) -> None:
    """Fija todas las semillas relevantes para que el entrenamiento sea
    reproducible corrida a corrida."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE en la escala original de la variable (quetzales/dólares)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_csv(path) -> pd.DataFrame:
    """Carga un CSV del proyecto normalizando las diferencias de formato entre
    ``train.csv`` y el archivo de prueba del profesor.

    Diferencias observadas en ``pipeline_test.csv`` que hay que absorber aquí:

    1. Valores categóricos envueltos en comillas simples (``'Wd Shng'``), que de
       no limpiarse crearían una categoría distinta a la vista en entrenamiento.
    2. Columnas numéricas que vienen como entero en el test y float en train.
    3. Celdas vacías vs. literal ``NA``.
    """
    df = pd.read_csv(path, keep_default_na=True, na_values=["", "NA", "N/A", "nan"])

    # 1. Quitar comillas simples/dobles residuales y espacios en los strings.
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.strip("'\"")
            .str.strip()
            .replace({"nan": np.nan, "": np.nan, "None": np.nan})
        )

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(int)

    return df


def split_features_target(df: pd.DataFrame):
    """Separa (X, y, ids). ``y`` es None cuando el CSV no trae SalePrice."""
    ids = df[ID_COL].to_numpy() if ID_COL in df.columns else np.arange(len(df))
    y = df[TARGET].to_numpy(dtype=float) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    return X, y, ids
