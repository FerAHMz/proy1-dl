"""Etapa 2: el historial de iteraciones (sección 2.3 del escrito).

Cambio una sola cosa por iteración respecto a alguna anterior, para poder decir
qué produjo la mejora en vez de quedarme con una combinación que funcionó por
razones que no sé. Todas las corro con el mismo protocolo (5-fold sobre
desarrollo) y la misma semilla.

Salidas: reports/experiments.csv y reports/figures/07_curvas_*.png

Uso:  python src/02_experiments.py [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EXPERIMENTS_CSV, FIGURES, TRAIN_CSV
from experiment import cross_validate, make_holdout
from model import HParams
from utils import load_csv, set_seed

# (id, qué cambié respecto a la iteración anterior, hiperparámetros)
EXPERIMENTS = [
    ("it01", "Baseline: 1 capa oculta (64), sin regularización, target crudo",
     HParams(hidden=(64,), dropout=0.0, batch_norm=False, log_target=False,
             weight_decay=0.0, loss="mse", scheduler="none", epochs=300)),

    ("it02", "it01 + target en log1p(SalePrice)",
     HParams(hidden=(64,), dropout=0.0, batch_norm=False, log_target=True,
             weight_decay=0.0, loss="mse", scheduler="none", epochs=300)),

    ("it03", "it02 + red más profunda (256-128-64)",
     HParams(hidden=(256, 128, 64), dropout=0.0, batch_norm=False,
             log_target=True, weight_decay=0.0, loss="mse", scheduler="none",
             epochs=300)),

    ("it04", "it03 + BatchNorm",
     HParams(hidden=(256, 128, 64), dropout=0.0, batch_norm=True,
             log_target=True, weight_decay=0.0, loss="mse", scheduler="none",
             epochs=300)),

    ("it05", "it04 + Dropout 0.2",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=0.0, loss="mse", scheduler="none",
             epochs=300)),

    ("it06", "it05 + weight decay 1e-4 (AdamW)",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="mse", scheduler="none",
             epochs=300)),

    ("it07", "it06 + scheduler coseno",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="mse",
             scheduler="cosine", epochs=400)),

    ("it08", "it07 + pérdida Huber en vez de MSE",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber",
             scheduler="cosine", epochs=400)),

    ("it09", "it08 con red más ancha (512-256-128)",
     HParams(hidden=(512, 256, 128), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber",
             scheduler="cosine", epochs=400)),

    ("it10", "it08 con red más angosta (128-64)",
     HParams(hidden=(128, 64), dropout=0.2, batch_norm=True, log_target=True,
             weight_decay=1e-4, loss="huber", scheduler="cosine", epochs=400)),

    ("it11", "it08 + dropout más agresivo (0.35)",
     HParams(hidden=(256, 128, 64), dropout=0.35, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber",
             scheduler="cosine", epochs=400)),

    ("it12", "it08 + dropout más suave (0.1)",
     HParams(hidden=(256, 128, 64), dropout=0.1, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber",
             scheduler="cosine", epochs=400)),

    ("it13", "it08 + weight decay más fuerte (1e-3)",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-3, loss="huber",
             scheduler="cosine", epochs=400)),

    ("it14", "it08 + lr más alto (3e-3)",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber", lr=3e-3,
             scheduler="cosine", epochs=400)),

    ("it15", "it08 + batch más pequeño (32)",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber", batch_size=32,
             scheduler="cosine", epochs=400)),

    ("it16", "it08 SIN remover outliers (control del preprocesamiento)",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True,
             log_target=True, weight_decay=1e-4, loss="huber",
             scheduler="cosine", epochs=400)),
]

NO_OUTLIER_REMOVAL = {"it16"}


def plot_curves(exp_id: str, desc: str, histories: list[dict]) -> None:
    """RMSE contra época del primer fold. Es la forma más rápida de ver si hay
    overfitting (val sube mientras train baja) o underfitting (las dos se quedan
    estancadas arriba)."""
    h = histories[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(h["epoch"], h["train_rmse"], label="train", lw=1.6)
    ax.plot(h["epoch"], h["val_rmse"], label="validación", lw=1.6)
    best = int(min(range(len(h["val_rmse"])), key=lambda i: h["val_rmse"][i]))
    ax.axvline(best, ls="--", c="gray", lw=1)
    ax.annotate(f"mejor época {best}\nval={h['val_rmse'][best]:,.0f}",
                xy=(best, h["val_rmse"][best]), xytext=(8, 14),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("época")
    ax.set_ylabel("RMSE (USD)")
    ax.set_title(f"{exp_id}: {desc}", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"07_curvas_{exp_id}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    quick = "--quick" in sys.argv
    set_seed()

    df = load_csv(TRAIN_CSV)
    df_dev, df_hold = make_holdout(df)
    print(f"Desarrollo: {len(df_dev)} filas | Holdout (intacto): {len(df_hold)} filas\n")

    experiments = EXPERIMENTS[:3] if quick else EXPERIMENTS
    rows = []

    for exp_id, desc, hp in experiments:
        print(f"[{exp_id}] {desc}")
        res = cross_validate(
            df_dev, hp,
            remove_outliers=exp_id not in NO_OUTLIER_REMOVAL,
        )
        print(f"  CV RMSE = {res['cv_rmse_mean']:,.0f} ± {res['cv_rmse_std']:,.0f}"
              f" | train {res['train_rmse_mean']:,.0f}"
              f" | gap {res['gap']:,.0f} | {res['seconds']}s\n")

        plot_curves(exp_id, desc, res["histories"])
        rows.append({
            "iteracion": exp_id,
            "cambio": desc,
            "cv_rmse_mean": round(res["cv_rmse_mean"], 1),
            "cv_rmse_std": round(res["cv_rmse_std"], 1),
            "train_rmse_mean": round(res["train_rmse_mean"], 1),
            "gap_val_menos_train": round(res["gap"], 1),
            "oof_rmse": round(res["oof_rmse"], 1),
            "mean_best_epoch": res["mean_best_epoch"],
            "segundos": res["seconds"],
            **hp.to_row(),
        })

    out = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    out.to_csv(EXPERIMENTS_CSV, index=False)

    print("=" * 78)
    print("Ranking de iteraciones por RMSE de validación cruzada:")
    print(out[["iteracion", "cv_rmse_mean", "cv_rmse_std",
               "train_rmse_mean", "gap_val_menos_train", "cambio"]]
          .to_string(index=False))
    print(f"\nTabla completa -> {EXPERIMENTS_CSV}")


if __name__ == "__main__":
    main()
