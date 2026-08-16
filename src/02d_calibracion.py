"""Etapa 2d: calibro la predicción pensando en la métrica de la competencia.

Son dos ajustes que no tocan la red, solo cómo convierto su salida a precio.

1) SMEARING (Duan, 1983).
   Como entreno sobre log1p(precio), `expm1(pred)` me da la mediana condicional
   y no la media. Pero el RMSE se minimiza con la media, y en una variable con
   cola derecha la media queda arriba de la mediana, así que convertir con
   expm1 a secas me subestima siempre.

   El estimador de smearing lo corrige multiplicando por el promedio de los
   residuos exponenciados del entrenamiento:

       factor = mean(exp(residuo_log_entrenamiento))
       precio = expm1(pred_log) * factor

   El factor lo calculo solo con los datos de entrenamiento de cada fold.

2) BARRIDO DEL TECHO.
   El techo terminó siendo lo que más impacto tuvo en 02c_robustez.py, así que
   acá elijo su cuantil con validación cruzada en vez de dejarlo puesto a ojo.
   Todo sobre el bloque de desarrollo; el holdout no entra.

Uso:  python src/02d_calibracion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import REPORTS, TARGET, TRAIN_CSV
from experiment import drop_outliers, make_holdout, stratify_bins
from model import HParams, predict, train_model
from preprocessing import Preprocessor
from utils import load_csv, rmse, set_seed, split_features_target

REPEAT_SEEDS = [42, 7, 2024]

HP = HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True, log_target=True,
             weight_decay=1e-4, loss="huber", scheduler="cosine", epochs=400,
             es_metric="log_rmse")

CEIL_GRID = [(0.98, 1.0), (0.99, 1.0), (0.995, 1.0), (0.995, 1.10),
             (0.999, 1.10), (1.0, 1.10), (1.0, 1.30)]


def collect_predictions(df_dev, seed):
    """Corro la CV una sola vez y me guardo las predicciones crudas y el factor
    de smearing de cada fold. Así puedo barrer techo y smearing sin volver a
    entrenar la red para cada combinación."""
    y_all = df_dev[TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof_raw = np.zeros(len(df_dev))
    smear = np.zeros(len(df_dev))
    train_max = np.zeros(len(df_dev))
    train_min = np.zeros(len(df_dev))
    ytr_store = []

    for tr_i, va_i in kf.split(np.arange(len(df_dev)), stratify_bins(y_all)):
        df_tr = drop_outliers(df_dev.iloc[tr_i].reset_index(drop=True))
        df_va = df_dev.iloc[va_i].reset_index(drop=True)

        Xtr_raw, ytr, _ = split_features_target(df_tr)
        Xva_raw, yva, _ = split_features_target(df_va)

        pre = Preprocessor()
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)

        res = train_model(Xtr, ytr, Xva, yva, HP)

        # El factor sale de los residuos de entrenamiento, nunca de validación.
        pred_tr = predict(res.model, Xtr)
        resid_log = np.log1p(ytr) - np.log1p(np.clip(pred_tr, 0, None))
        smear[va_i] = float(np.mean(np.exp(resid_log)))

        oof_raw[va_i] = predict(res.model, Xva)
        train_max[va_i] = ytr.max()
        train_min[va_i] = ytr.min()
        ytr_store.append(ytr)

    return y_all, oof_raw, smear, train_min, ytr_store, kf


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    df_dev, _ = make_holdout(df)

    print("Corriendo CV para recolectar predicciones crudas...\n")
    runs = []
    for s in REPEAT_SEEDS:
        y_all, oof_raw, smear, train_min, ytr_store, kf = collect_predictions(df_dev, s)
        runs.append((y_all, oof_raw, smear, train_min, ytr_store, kf))
        print(f"  semilla {s}: RMSE crudo sin calibrar = {rmse(y_all, oof_raw):,.0f}"
              f" | factor smearing medio = {smear.mean():.4f}")

    print("\nFactor de smearing > 1 confirma que expm1 subestima el precio medio.\n")

    rows = []
    for use_smear in [False, True]:
        for q, margin in CEIL_GRID:
            scores = []
            for y_all, oof_raw, smear, train_min, ytr_store, kf in runs:
                pred = oof_raw * smear if use_smear else oof_raw.copy()

                # El techo de cada fold sale del y de entrenamiento de ese fold.
                capped = pred.copy()
                for (tr_i, va_i), ytr in zip(
                        kf.split(np.arange(len(y_all)), stratify_bins(y_all)),
                        ytr_store):
                    ceil = float(np.quantile(ytr, q)) * margin
                    capped[va_i] = np.clip(pred[va_i], ytr.min() * 0.5, ceil)

                scores.append(rmse(y_all, capped))

            rows.append({
                "smearing": use_smear,
                "cuantil_techo": q,
                "margen": margin,
                "rmse": round(float(np.mean(scores)), 1),
                "std": round(float(np.std(scores)), 1),
            })

    out = pd.DataFrame(rows).sort_values("rmse")
    out.to_csv(REPORTS / "calibracion.csv", index=False)

    print("=" * 70)
    print(out.to_string(index=False))

    best = out.iloc[0]
    sin_cal = out[(~out.smearing) & (out.cuantil_techo == 1.0)
                  & (out.margen == 1.30)].rmse.iloc[0]
    print(f"\nMejor combinación: smearing={best.smearing}, "
          f"techo=q{best.cuantil_techo}x{best.margen} -> {best.rmse:,.0f} USD")
    print(f"Referencia casi sin acotar:                     {sin_cal:,.0f} USD")
    print(f"Ganancia por calibración:                       {sin_cal - best.rmse:,.0f} USD")
    print(f"\nTabla -> {REPORTS / 'calibracion.csv'}")


if __name__ == "__main__":
    main()
