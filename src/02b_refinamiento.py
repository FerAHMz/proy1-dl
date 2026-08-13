"""Etapa 2b: desempate entre las mejores configuraciones con CV repetida.

Motivo: en la primera ronda la desviación entre folds (~9,000–17,000 USD) resultó
MAYOR que las diferencias entre las mejores configuraciones (~900 USD). Con una
sola corrida de 5-fold no se puede distinguir cuál generaliza mejor: elegir el
menor RMSE puntual sería elegir ruido del split.

Solución: repetir la validación cruzada con varias semillas (3 x 5 folds = 15
estimaciones por configuración) y comparar por la media de esas repeticiones y
por el gap train/val, no por un número suelto.

Uso:  python src/02b_refinamiento.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import REPORTS, TRAIN_CSV
from experiment import cross_validate, make_holdout
from model import HParams
from utils import load_csv, set_seed

REPEAT_SEEDS = [42, 7, 2024]

# Candidatos: las mejores de la ronda 1 más variantes de regularización en torno
# a ellas. Todas comparten arquitectura, porque la ronda 1 mostró que ni
# ensanchar ni angostar la red aporta; lo que decide es la regularización.
CANDIDATES = [
    ("c1", "it05: BN + dropout 0.2, MSE, sin scheduler",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True, log_target=True,
             weight_decay=0.0, loss="mse", scheduler="none", epochs=300)),

    ("c2", "it08: BN + dropout 0.2 + wd 1e-4, Huber, coseno",
     HParams(hidden=(256, 128, 64), dropout=0.2, batch_norm=True, log_target=True,
             weight_decay=1e-4, loss="huber", scheduler="cosine", epochs=400)),

    ("c3", "it11: como c2 pero dropout 0.35",
     HParams(hidden=(256, 128, 64), dropout=0.35, batch_norm=True, log_target=True,
             weight_decay=1e-4, loss="huber", scheduler="cosine", epochs=400)),

    ("c4", "c2 con dropout intermedio 0.25 + wd 5e-4",
     HParams(hidden=(256, 128, 64), dropout=0.25, batch_norm=True, log_target=True,
             weight_decay=5e-4, loss="huber", scheduler="cosine", epochs=400)),

    ("c5", "c2 con dropout 0.3 + wd 1e-3 (regularización fuerte)",
     HParams(hidden=(256, 128, 64), dropout=0.3, batch_norm=True, log_target=True,
             weight_decay=1e-3, loss="huber", scheduler="cosine", epochs=400)),

    ("c6", "c4 con más paciencia y coseno largo (600 épocas)",
     HParams(hidden=(256, 128, 64), dropout=0.25, batch_norm=True, log_target=True,
             weight_decay=5e-4, loss="huber", scheduler="cosine", epochs=600,
             patience=100)),
]


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    df_dev, _ = make_holdout(df)

    print(f"CV repetida: {len(REPEAT_SEEDS)} semillas x 5 folds "
          f"= {len(REPEAT_SEEDS) * 5} estimaciones por candidato\n")

    rows = []
    for cid, desc, hp in CANDIDATES:
        vals, trains, gaps = [], [], []
        for s in REPEAT_SEEDS:
            r = cross_validate(df_dev, hp, seed=s)
            vals.append(r["cv_rmse_mean"])
            trains.append(r["train_rmse_mean"])
            gaps.append(r["gap"])

        mean_val = float(np.mean(vals))
        # Desviación ENTRE repeticiones: mide qué tan reproducible es el
        # resultado, no la dispersión interna de los folds.
        std_rep = float(np.std(vals))
        rows.append({
            "candidato": cid,
            "descripcion": desc,
            "cv_rmse_mean": round(mean_val, 1),
            "std_entre_repeticiones": round(std_rep, 1),
            "train_rmse_mean": round(float(np.mean(trains)), 1),
            "gap": round(float(np.mean(gaps)), 1),
            "peor_repeticion": round(float(np.max(vals)), 1),
            "arquitectura": "-".join(map(str, hp.hidden)),
            "dropout": hp.dropout,
            "weight_decay": hp.weight_decay,
            "loss": hp.loss,
            "epochs": hp.epochs,
        })
        print(f"[{cid}] {desc}")
        print(f"  CV RMSE {mean_val:>9,.0f} ± {std_rep:>6,.0f} (entre repeticiones)"
              f" | train {np.mean(trains):>9,.0f} | gap {np.mean(gaps):>8,.0f}"
              f" | peor {np.max(vals):>9,.0f}\n")

    out = pd.DataFrame(rows).sort_values("cv_rmse_mean")
    out.to_csv(REPORTS / "refinamiento.csv", index=False)

    print("=" * 100)
    print(out[["candidato", "cv_rmse_mean", "std_entre_repeticiones",
               "train_rmse_mean", "gap", "peor_repeticion", "descripcion"]]
          .to_string(index=False))

    best = out.iloc[0]
    print(f"\nMejor por RMSE medio: {best.candidato} ({best.cv_rmse_mean:,.0f})")
    print(f"Menor gap:            {out.sort_values('gap').iloc[0].candidato}")
    print(f"Más estable:          {out.sort_values('std_entre_repeticiones').iloc[0].candidato}")
    print(f"\nTabla -> {REPORTS / 'refinamiento.csv'}")


if __name__ == "__main__":
    main()
