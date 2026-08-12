"""Protocolo de evaluación compartido por todos los experimentos.

Regla del proyecto: el holdout se aparta UNA vez y no se toca hasta el final.
Toda decisión (arquitectura, hiperparámetros, features) se toma con la
validación cruzada sobre el bloque de desarrollo. Así el RMSE del holdout es
una estimación honesta de lo que hará el modelo con el dataset de prueba.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from config import SEED, TARGET
from model import HParams, TrainResult, predict, train_model
from preprocessing import Preprocessor
from utils import rmse

HOLDOUT_FRAC = 0.15
N_FOLDS = 5


def stratify_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bins por cuantil del precio: permite que cada split tenga una mezcla
    parecida de casas baratas y caras (si no, un fold puede quedarse con casi
    todas las casas de $500k+ y su RMSE deja de ser comparable)."""
    ranks = pd.Series(y).rank(method="first")
    return pd.qcut(ranks, q=n_bins, labels=False).to_numpy()


def make_holdout(df: pd.DataFrame, seed: int = SEED):
    """Divide train.csv en (desarrollo, holdout) de forma estratificada."""
    y = df[TARGET].to_numpy(dtype=float)
    dev_idx, hold_idx = train_test_split(
        np.arange(len(df)),
        test_size=HOLDOUT_FRAC,
        random_state=seed,
        stratify=stratify_bins(y),
    )
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[hold_idx].reset_index(drop=True)


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las ventas parciales anómalas (casas enormes a precio bajo).

    Solo se aplica al bloque de ENTRENAMIENTO, nunca a validación ni holdout:
    quitar casos difíciles de la evaluación inflaría artificialmente la métrica.
    """
    mask = (df["GrLivArea"] > 4000) & (df[TARGET] < 300000)
    return df.loc[~mask].reset_index(drop=True)


def cross_validate(
    df_dev: pd.DataFrame,
    hp: HParams,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    remove_outliers: bool = True,
    verbose: bool = False,
) -> dict:
    """K-fold sobre el bloque de desarrollo.

    Crítico: el ``Preprocessor`` se ajusta DENTRO de cada fold, solo con los
    datos de entrenamiento de ese fold. Ajustarlo antes del split filtraría
    medias, medianas y categorías del fold de validación (data leakage) y daría
    un RMSE optimista que no se sostiene en el dataset de prueba.
    """
    from utils import split_features_target

    y_all = df_dev[TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    bins = stratify_bins(y_all)

    fold_val, fold_train, fold_epochs, histories = [], [], [], []
    oof = np.zeros(len(df_dev))
    t0 = time.time()

    for k, (tr_i, va_i) in enumerate(kf.split(np.arange(len(df_dev)), bins)):
        df_tr = df_dev.iloc[tr_i].reset_index(drop=True)
        df_va = df_dev.iloc[va_i].reset_index(drop=True)
        if remove_outliers:
            df_tr = drop_outliers(df_tr)

        Xtr_raw, ytr, _ = split_features_target(df_tr)
        Xva_raw, yva, _ = split_features_target(df_va)

        pre = Preprocessor()
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)

        res: TrainResult = train_model(Xtr, ytr, Xva, yva, hp, verbose=verbose)
        oof[va_i] = predict(res.model, Xva)

        fold_val.append(res.best_val_rmse)
        fold_train.append(res.best_train_rmse)
        fold_epochs.append(res.best_epoch)
        histories.append(res.history)
        if verbose:
            print(f"  fold {k+1}/{n_folds}: val RMSE {res.best_val_rmse:,.0f} "
                  f"(train {res.best_train_rmse:,.0f}, época {res.best_epoch})")

    return {
        "cv_rmse_mean": float(np.mean(fold_val)),
        "cv_rmse_std": float(np.std(fold_val)),
        "train_rmse_mean": float(np.mean(fold_train)),
        "oof_rmse": rmse(y_all, oof),
        "gap": float(np.mean(fold_val) - np.mean(fold_train)),
        "mean_best_epoch": float(np.mean(fold_epochs)),
        "seconds": round(time.time() - t0, 1),
        "fold_val_rmse": fold_val,
        "histories": histories,
        "oof": oof,
    }
