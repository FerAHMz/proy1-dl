"""Modelo de la competencia (17 de agosto): ensemble diverso sobre todo train.

Fue el mejor de mis 14 envíos al leaderboard: RMSE 24,833 en el dataset de
prueba real. Probé además reentrenar con más datos, pseudo-labeling y target
encoding de vecindario; ninguno lo superó (todos quedaron entre 24.9k y 26k),
así que este quedó como el modelo final de competencia.

La lógica: promediar modelos casi iguales solo quita ruido de inicialización;
promediar modelos DISTINTOS (ancho, angosto, más regularizado, otra activación)
también promedia sus sesgos. Cada configuración se entrena con 10-fold para que
todo miembro tenga early stopping honesto en datos que no vio, y con las 1168
filas de train.csv (ya sin holdout: la configuración estaba decidida y
reservarlo solo quitaba datos).

Escribe submission_competencia.csv con el promedio uniforme de las 4
configuraciones (40 redes).

Uso:  python src/04_competencia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import ID_COL, SEED, TARGET, TRAIN_CSV
from experiment import drop_outliers, stratify_bins
from model import HParams, predict, train_model
from preprocessing import Preprocessor
from utils import load_csv, rmse, set_seed, split_features_target

N_FOLDS = 10

CONFIGS = {
    "base": HParams(hidden=(256, 128, 64), dropout=0.2, weight_decay=1e-4,
                    loss="huber", log_target=True, scheduler="cosine",
                    epochs=400, patience=60, es_metric="log_rmse"),
    "ancha": HParams(hidden=(512, 256, 128), dropout=0.3, weight_decay=5e-4,
                     loss="huber", log_target=True, scheduler="cosine",
                     epochs=400, patience=60, es_metric="log_rmse"),
    "angosta": HParams(hidden=(128, 64), dropout=0.15, weight_decay=1e-4,
                       loss="huber", log_target=True, scheduler="cosine",
                       epochs=400, patience=60, es_metric="log_rmse"),
    "silu": HParams(hidden=(256, 128, 64), activation="silu", dropout=0.25,
                    weight_decay=3e-4, loss="huber", log_target=True,
                    scheduler="cosine", epochs=400, patience=60,
                    es_metric="log_rmse"),
}


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    y_all = df[TARGET].to_numpy(dtype=float)
    test = load_csv("data/raw/test_features.csv")
    X_test = test.drop(columns=[c for c in (ID_COL, TARGET) if c in test.columns])

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    bins = stratify_bins(y_all)
    splits = list(kf.split(np.arange(len(df)), bins))

    oof = {n: np.zeros(len(df)) for n in CONFIGS}
    test_pred = {n: [] for n in CONFIGS}

    for k, (tr_i, va_i) in enumerate(splits):
        df_tr = drop_outliers(df.iloc[tr_i].reset_index(drop=True))
        df_va = df.iloc[va_i].reset_index(drop=True)
        Xtr_raw, ytr, _ = split_features_target(df_tr)
        Xva_raw, yva, _ = split_features_target(df_va)
        pre = Preprocessor()
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)
        Xte = pre.transform(X_test)

        for name, hp in CONFIGS.items():
            set_seed(SEED + k * 10)
            res = train_model(Xtr, ytr, Xva, yva, hp)
            pred_tr = predict(res.model, Xtr)
            sm = float(np.mean(np.exp(np.log1p(ytr) - np.log1p(np.clip(pred_tr, 0, None)))))
            oof[name][va_i] = predict(res.model, Xva) * sm
            test_pred[name].append(predict(res.model, Xte) * sm)
        print(f"fold {k + 1:2d}/{N_FOLDS} listo")

    print("\nOOF por configuración:")
    for n in CONFIGS:
        print(f"  {n:8s}: RMSE {rmse(y_all, oof[n]):>9,.0f}"
              f" | log-RMSE {rmse(np.log1p(y_all), np.log1p(oof[n])):.4f}")

    # Predicción final: promedio uniforme de las 4 configuraciones.
    names = list(CONFIGS)
    stack_test = np.vstack([np.mean(test_pred[n], axis=0) for n in names])
    uni_oof = np.vstack([oof[n] for n in names]).mean(axis=0)
    print(f"\nBlend uniforme OOF: {rmse(y_all, uni_oof):,.0f}"
          f" | log-RMSE {rmse(np.log1p(y_all), np.log1p(np.clip(uni_oof, 1, None))):.4f}")

    out = np.clip(stack_test.mean(axis=0), 30000, None)
    pd.DataFrame({ID_COL: test[ID_COL], "Prediction": np.round(out, 2)}) \
        .to_csv("submission_competencia.csv", index=False)
    print(f"submission_competencia.csv: max {out.max():,.0f} media {out.mean():,.0f}")


if __name__ == "__main__":
    main()
