"""Etapa 2c: corrección de las tres decisiones que el diagnóstico puso en duda.

Contexto. El primer entrenamiento final dio un RMSE de holdout de 39,679 contra
30,205 de validación cruzada. Al abrir los residuos, UN solo caso (Id 524: casa
de calidad 10 y 4,676 sq ft vendida sin terminar a 184,750) aportaba el 82% del
error cuadrático total. Sin él, el RMSE del holdout bajaba a 17,080.

Eso destapó tres decisiones mal tomadas:

  A. El early stopping seleccionaba el checkpoint por RMSE en escala original.
     Esa métrica la dominan una o dos casas extremas por fold, así que como
     señal de selección es ruido: hubo miembros que se detuvieron en la época 2.
     -> Se compara contra seleccionar por RMSE en escala logarítmica.

  B. Se removían los 2 outliers de venta parcial del entrenamiento. El efecto
     colateral es que el modelo nunca aprende que `SaleCondition=Partial` con
     casa grande se vende por debajo de su valor — información que SÍ está en
     las columnas. Por eso predijo 638,804 para el Id 524.
     -> Se compara removerlos contra conservarlos.

  C. No había techo de predicción. Predecir 638,804 cuando solo 19 de 992 casas
     superan los 400,000 es extrapolación agresiva, y en RMSE se paga al
     cuadrado.
     -> Se compara sin techo contra un techo aprendido del entrenamiento.

Cada combinación se evalúa con CV repetida (3 semillas x 5 folds).

Uso:  python src/02c_robustez.py
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

BASE = dict(hidden=(256, 128, 64), dropout=0.2, batch_norm=True, log_target=True,
            weight_decay=1e-4, loss="huber", scheduler="cosine", epochs=400)

# (id, descripción, es_metric, remover outliers, aplicar techo)
VARIANTS = [
    ("v1", "Original: ES por RMSE, remueve outliers, sin techo", "rmse", True, False),
    ("v2", "ES por log-RMSE, remueve outliers, sin techo", "log_rmse", True, False),
    ("v3", "ES por log-RMSE, CONSERVA outliers, sin techo", "log_rmse", False, False),
    ("v4", "ES por log-RMSE, remueve outliers, CON techo", "log_rmse", True, True),
    ("v5", "ES por log-RMSE, CONSERVA outliers, CON techo", "log_rmse", False, True),
]

# Techo: percentil 99.5 del precio de entrenamiento, con 10% de margen para no
# truncar casas legítimamente caras. Se aprende del train de cada fold, nunca
# del conjunto de validación.
CEIL_QUANTILE = 0.995
CEIL_MARGIN = 1.10


def evaluate(df_dev, es_metric, remove_out, use_ceiling, seed) -> dict:
    hp = HParams(**BASE, es_metric=es_metric)
    y_all = df_dev[TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof = np.zeros(len(df_dev))
    trains, epochs = [], []

    for tr_i, va_i in kf.split(np.arange(len(df_dev)), stratify_bins(y_all)):
        df_tr = df_dev.iloc[tr_i].reset_index(drop=True)
        df_va = df_dev.iloc[va_i].reset_index(drop=True)
        if remove_out:
            df_tr = drop_outliers(df_tr)

        Xtr_raw, ytr, _ = split_features_target(df_tr)
        Xva_raw, yva, _ = split_features_target(df_va)

        pre = Preprocessor()
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)

        res = train_model(Xtr, ytr, Xva, yva, hp)
        pred = predict(res.model, Xva)

        if use_ceiling:
            ceiling = float(np.quantile(ytr, CEIL_QUANTILE)) * CEIL_MARGIN
            pred = np.clip(pred, ytr.min() * 0.5, ceiling)

        oof[va_i] = pred
        trains.append(res.best_train_rmse)
        epochs.append(res.best_epoch)

    residual = y_all - oof
    worst = np.argsort(np.abs(residual))[-5:]
    return {
        "oof_rmse": rmse(y_all, oof),
        # RMSE recortado: cuánto del error viene del grueso de los datos y no de
        # los pocos casos extremos.
        "oof_rmse_sin_top5": rmse(np.delete(y_all, worst), np.delete(oof, worst)),
        "mae": float(np.mean(np.abs(residual))),
        "log_rmse": rmse(np.log1p(y_all), np.log1p(np.clip(oof, 0, None))),
        "train_rmse": float(np.mean(trains)),
        "mean_epoch": float(np.mean(epochs)),
        "max_pred": float(oof.max()),
    }


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    df_dev, _ = make_holdout(df)

    print(f"CV repetida: {len(REPEAT_SEEDS)} semillas x 5 folds por variante\n")
    rows = []

    for vid, desc, es_metric, remove_out, ceiling in VARIANTS:
        runs = [evaluate(df_dev, es_metric, remove_out, ceiling, s)
                for s in REPEAT_SEEDS]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        agg["std_oof_rmse"] = float(np.std([r["oof_rmse"] for r in runs]))

        rows.append({"variante": vid, "descripcion": desc,
                     "es_metric": es_metric, "remueve_outliers": remove_out,
                     "techo": ceiling,
                     **{k: round(v, 1) for k, v in agg.items()}})

        print(f"[{vid}] {desc}")
        print(f"  OOF RMSE {agg['oof_rmse']:>9,.0f} ± {agg['std_oof_rmse']:>6,.0f}"
              f" | sin top-5 {agg['oof_rmse_sin_top5']:>8,.0f}"
              f" | MAE {agg['mae']:>8,.0f}"
              f" | log-RMSE {agg['log_rmse']:.4f}")
        print(f"  train {agg['train_rmse']:>9,.0f} | gap {agg['oof_rmse']-agg['train_rmse']:>8,.0f}"
              f" | época media {agg['mean_epoch']:>5.0f}"
              f" | pred máx {agg['max_pred']:>9,.0f}\n")

    out = pd.DataFrame(rows).sort_values("oof_rmse")
    out.to_csv(REPORTS / "robustez.csv", index=False)

    print("=" * 104)
    print(out[["variante", "oof_rmse", "std_oof_rmse", "oof_rmse_sin_top5",
               "mae", "train_rmse", "mean_epoch", "descripcion"]].to_string(index=False))
    print(f"\nGanador: {out.iloc[0].variante} — {out.iloc[0].descripcion}")
    print(f"Tabla -> {REPORTS / 'robustez.csv'}")


if __name__ == "__main__":
    main()
