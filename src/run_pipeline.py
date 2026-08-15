"""Orquestador: corre el pipeline completo de cero.

    python src/run_pipeline.py            # todo
    python src/run_pipeline.py --skip-exp # solo EDA y modelo final

Las etapas de experimentación (02, 02b, 02c, 02d) son las más lentas (~25 min en
total) y solo hace falta correrlas para reproducir las tablas del trabajo
escrito. Para regenerar el modelo entrenado bastan la 01 y la 03.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent

ETAPAS = [
    ("01_eda.py", "Análisis exploratorio", False),
    ("02_experiments.py", "Historial de 16 iteraciones", True),
    ("02b_refinamiento.py", "Desempate con CV repetida", True),
    ("02c_robustez.py", "Corrección de early stopping, outliers y techo", True),
    ("02d_calibracion.py", "Calibración: techo y smearing", True),
    ("03_train_final.py", "Entrenamiento del modelo final + holdout", False),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-exp", action="store_true",
                    help="omitir las etapas de experimentación")
    args = ap.parse_args()

    t0 = time.time()
    for script, desc, es_experimento in ETAPAS:
        if es_experimento and args.skip_exp:
            print(f"[omitida] {script} — {desc}")
            continue

        print(f"\n{'=' * 70}\n[{script}] {desc}\n{'=' * 70}")
        r = subprocess.run([sys.executable, str(SRC / script)])
        if r.returncode != 0:
            print(f"\nFalló {script} (código {r.returncode})")
            sys.exit(r.returncode)

    print(f"\n{'=' * 70}")
    print(f"Pipeline completo en {(time.time() - t0) / 60:.1f} min")
    print("Predecir sobre un dataset nuevo:")
    print("  python src/predict.py --input <archivo.csv> --output submission.csv")


if __name__ == "__main__":
    main()
