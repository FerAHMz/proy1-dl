"""Etapa 3: entreno el modelo final y guardo todo lo necesario para predecir.

El modelo final no es una red sino un ensemble: uno por fold de la validación
cruzada. Promediarlos me quita la varianza que meten la inicialización y el
split, sin que la cosa memorice más: cada miembro vio solo el 80% de los datos
y se detuvo por early stopping en su propio fold.

Deja en models/:
  - preprocessor.pkl  los números del preprocesamiento (medianas, categorías,
                      medias y desviaciones)
  - mlp.pt            pesos de cada miembro, arquitectura e hiperparámetros

Uso:  python src/03_train_final.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from config import (
    CEIL_MARGIN,
    CEIL_QUANTILE,
    EXPERIMENTS_CSV,
    FLOOR_FACTOR,
    MODEL_PT,
    PREPROCESSOR_PKL,
    REPORTS,
    SEED,
    TARGET,
)
from experiment import N_FOLDS, drop_outliers, make_holdout, stratify_bins
from model import HParams, predict, train_model
from preprocessing import Preprocessor
from utils import load_csv, rmse, set_seed, split_features_target
from config import TRAIN_CSV

# La configuración final la dejo escrita acá en vez de leerla de los CSV, para
# que el entrenamiento salga igual aunque vuelva a correr los experimentos.
#
# De dónde salió cada cosa:
#   - arquitectura y regularización -> candidato c2 de 02b_refinamiento.py
#   - es_metric="log_rmse"          -> variante v4 de 02c_robustez.py
#   - techo de predicción           -> variante v4 de 02c_robustez.py
BEST_HP = HParams(
    hidden=(256, 128, 64),
    activation="gelu",
    dropout=0.2,
    batch_norm=True,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=400,
    patience=60,
    loss="huber",
    log_target=True,
    scheduler="cosine",
    es_metric="log_rmse",
)

# Le doy una semilla distinta a cada miembro para que el ensemble varíe por
# algo más que el split.
ENSEMBLE_SEEDS = [SEED, SEED + 1, SEED + 2]


def load_best_hp_note() -> str:
    """Para que el log del entrenamiento me recuerde de dónde salió cada
    decisión."""
    partes = []
    for archivo, col_id, col_rmse, etapa in [
        (REPORTS / "refinamiento.csv", "candidato", "cv_rmse_mean", "arquitectura"),
        (REPORTS / "robustez.csv", "variante", "oof_rmse", "early stopping"),
        (REPORTS / "calibracion.csv", None, "rmse", "calibración"),
    ]:
        if archivo.exists():
            top = pd.read_csv(archivo).sort_values(col_rmse).iloc[0]
            nombre = top[col_id] if col_id else "techo+smearing"
            partes.append(f"{etapa}: {nombre} ({top[col_rmse]:,.0f})")
    return " | ".join(partes) if partes else "(configuración por defecto)"


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    df_dev, df_hold = make_holdout(df)

    print(f"Configuración final {load_best_hp_note()}")
    print(f"Desarrollo: {len(df_dev)} | Holdout: {len(df_hold)}\n")

    y_dev = df_dev[TARGET].to_numpy(dtype=float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    bins = stratify_bins(y_dev)

    # El rango de precios que considero posible, sacado solo de desarrollo.
    # Es lo que evita que la red se vaya lejos de lo que vio entrenando.
    ceiling = float(np.quantile(y_dev, CEIL_QUANTILE)) * CEIL_MARGIN
    floor = float(y_dev.min()) * FLOOR_FACTOR
    print(f"Rango de predicción permitido: {floor:,.0f} – {ceiling:,.0f} USD\n")

    members: list[dict] = []
    histories: list[dict] = []
    oof = np.zeros(len(df_dev))
    oof_counts = np.zeros(len(df_dev))

    for k, (tr_i, va_i) in enumerate(kf.split(np.arange(len(df_dev)), bins)):
        df_tr = drop_outliers(df_dev.iloc[tr_i].reset_index(drop=True))
        df_va = df_dev.iloc[va_i].reset_index(drop=True)

        Xtr_raw, ytr, _ = split_features_target(df_tr)
        Xva_raw, yva, _ = split_features_target(df_va)

        pre = Preprocessor()
        Xtr = pre.fit_transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)

        for s in ENSEMBLE_SEEDS:
            set_seed(s)
            res = train_model(Xtr, ytr, Xva, yva, BEST_HP)
            if s == ENSEMBLE_SEEDS[0]:
                histories.append(res.history)

            # Smearing (Duan): compensa que expm1 me da la mediana y no la
            # media. Lo calculo con los residuos de entrenamiento del miembro,
            # nunca con los de validación.
            pred_tr = predict(res.model, Xtr)
            resid_log = np.log1p(ytr) - np.log1p(np.clip(pred_tr, 0, None))
            smearing = float(np.mean(np.exp(resid_log)))

            members.append({
                "smearing": smearing,
                "state_dict": {k_: v.cpu() for k_, v in res.model.state_dict().items()},
                "target_mean": res.model.target_mean_,
                "target_std": res.model.target_std_,
                "log_target": res.model.log_target_,
                "preprocessor": pre,
                "fold": k,
                "seed": s,
                "val_rmse": res.best_val_rmse,
                "train_rmse": res.best_train_rmse,
                "best_epoch": res.best_epoch,
            })
            oof[va_i] += np.clip(predict(res.model, Xva) * smearing, floor, ceiling)
            oof_counts[va_i] += 1

        fold_members = [m for m in members if m["fold"] == k]
        print(f"fold {k+1}/{N_FOLDS}: val RMSE "
              f"{np.mean([m['val_rmse'] for m in fold_members]):,.0f} "
              f"| train {np.mean([m['train_rmse'] for m in fold_members]):,.0f} "
              f"| épocas {[m['best_epoch'] for m in fold_members]}")

    oof /= np.maximum(oof_counts, 1)
    oof_rmse = rmse(y_dev, oof)
    mean_val = float(np.mean([m["val_rmse"] for m in members]))
    mean_train = float(np.mean([m["train_rmse"] for m in members]))

    print(f"\nRMSE out-of-fold (ensemble): {oof_rmse:,.0f}")
    print(f"RMSE val promedio por miembro: {mean_val:,.0f}")
    print(f"RMSE train promedio: {mean_train:,.0f}  ->  gap {mean_val - mean_train:,.0f}")

    # --- Preprocesador final, reajustado con todo el bloque de desarrollo ---
    # Cada miembro usa el preprocesador de su propio fold, así que este lo dejo
    # nada más como referencia del pipeline.
    X_dev_raw, _, _ = split_features_target(drop_outliers(df_dev))
    pre_full = Preprocessor().fit(X_dev_raw)

    with open(PREPROCESSOR_PKL, "wb") as f:
        pickle.dump({"preprocessor": pre_full, "n_features": pre_full.n_features}, f)

    torch.save({
        "members": members,
        "hparams": BEST_HP,
        "n_features": members[0]["preprocessor"].n_features,
        "price_floor": floor,
        "price_ceiling": ceiling,
        "oof_rmse": oof_rmse,
        "mean_val_rmse": mean_val,
        "mean_train_rmse": mean_train,
        "seed": SEED,
    }, MODEL_PT)

    print(f"\nModelo  -> {MODEL_PT}  ({len(members)} miembros)")
    print(f"Preproc -> {PREPROCESSOR_PKL}")

    # --- Evaluación en el holdout ------------------------------------------
    # Primera y única vez que lo toco. Después de esto ya no ajusto nada.
    from predict import predict_dataframe

    y_hold = df_hold[TARGET].to_numpy(dtype=float)
    pred_hold = predict_dataframe(df_hold)
    hold_rmse = rmse(y_hold, pred_hold)

    print("\n" + "=" * 66)
    print(f"RMSE en HOLDOUT ({len(df_hold)} casas nunca usadas): {hold_rmse:,.0f}")
    print("=" * 66)

    pd.DataFrame({
        "Id": df_hold["Id"], "y_true": y_hold, "y_pred": pred_hold,
        "residual": y_hold - pred_hold,
    }).to_csv(REPORTS / "holdout_predictions.csv", index=False)

    pd.DataFrame([{
        "oof_rmse": round(oof_rmse, 1),
        "holdout_rmse": round(hold_rmse, 1),
        "mean_val_rmse": round(mean_val, 1),
        "mean_train_rmse": round(mean_train, 1),
        "gap": round(mean_val - mean_train, 1),
        "n_members": len(members),
    }]).to_csv(REPORTS / "final_metrics.csv", index=False)

    plot_final_curves(histories)


def plot_final_curves(histories: list[dict]) -> None:
    """Curvas de entrenamiento del modelo final, un panel por fold.

    Es la gráfica de over/underfitting que va en la sección 2.3. Marco la época
    donde el early stopping se quedó con los pesos.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config import FIGURES

    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for k, (ax, h) in enumerate(zip(axes, histories)):
        ax.plot(h["epoch"], h["train_rmse"], label="train", lw=1.5)
        ax.plot(h["epoch"], h["val_rmse"], label="validación", lw=1.5)
        best = int(np.argmin(h["val_rmse"]))
        ax.axvline(best, ls="--", c="gray", lw=1)
        ax.set_title(f"fold {k + 1} — mejor época {best}", fontsize=9)
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("RMSE (USD)")
            ax.legend(fontsize=8)

    fig.suptitle("Modelo final — curvas de entrenamiento por fold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "08_curvas_final.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
