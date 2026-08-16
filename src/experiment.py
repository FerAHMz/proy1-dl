"""El protocolo de evaluación que uso en todos los experimentos.

La regla que me puse: aparto el holdout una vez y no lo vuelvo a ver hasta el
final. Arquitectura, hiperparámetros y features los decido todos con la
validación cruzada sobre el bloque de desarrollo. Así el RMSE del holdout me
sirve de verdad para saber qué esperar del dataset de prueba.
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
    """Parte los precios en bins por cuantil para que todos los splits queden
    con una mezcla parecida de casas baratas y caras. Sin esto un fold se puede
    quedar con casi todas las de $500k+ y su RMSE ya no compara con nada."""
    ranks = pd.Series(y).rank(method="first")
    return pd.qcut(ranks, q=n_bins, labels=False).to_numpy()


def make_holdout(df: pd.DataFrame, seed: int = SEED):
    """Parte train.csv en (desarrollo, holdout) estratificando por precio."""
    y = df[TARGET].to_numpy(dtype=float)
    dev_idx, hold_idx = train_test_split(
        np.arange(len(df)),
        test_size=HOLDOUT_FRAC,
        random_state=seed,
        stratify=stratify_bins(y),
    )
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[hold_idx].reset_index(drop=True)


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Saca las ventas parciales raras: casas enormes a precio de ganga.

    Esto lo aplico solo a lo que entreno, nunca a validación ni al holdout. Si
    quitara los casos difíciles también de la evaluación, estaría inflando mi
    propia métrica.
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

    Lo importante acá: ajusto el ``Preprocessor`` dentro de cada fold, con los
    datos de entrenamiento de ese fold nada más. Si lo ajustara antes de partir,
    las medias, medianas y categorías del fold de validación se me colarían al
    de entrenamiento y el RMSE saldría optimista sin sostenerse después.
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
