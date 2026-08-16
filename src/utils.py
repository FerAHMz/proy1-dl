"""Cosas que uso en varios scripts: semillas, RMSE y la carga de los CSVs."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from config import ID_COL, SEED, TARGET


def set_seed(seed: int = SEED) -> None:
    """Fijo todas las semillas para que dos corridas den lo mismo."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE en dólares, la escala original del precio."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_csv(path) -> pd.DataFrame:
    """Carga un CSV emparejando el formato de train.csv con el del archivo de
    prueba.

    Comparando ambos encontré tres diferencias que resuelvo aquí:

    1. Categóricas con comillas simples (``'Wd Shng'``). Si no las limpio me
       quedan como una categoría distinta a la que vi entrenando.
    2. Columnas que en el test vienen enteras y en train float.
    3. Celdas vacías contra el literal ``NA``.
    """
    df = pd.read_csv(path, keep_default_na=True, na_values=["", "NA", "N/A", "nan"])

    # Quito comillas y espacios sobrantes de todos los strings.
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
    """Separa (X, y, ids). Devuelve y=None si el CSV no trae SalePrice."""
    ids = df[ID_COL].to_numpy() if ID_COL in df.columns else np.arange(len(df))
    y = df[TARGET].to_numpy(dtype=float) if TARGET in df.columns else None
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    return X, y, ids
