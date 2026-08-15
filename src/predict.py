"""Predicción sobre un dataset nuevo — script del día de la presentación.

Carga los artefactos entrenados, aplica EXACTAMENTE el mismo preprocesamiento
del entrenamiento y escribe un CSV con el formato de expected_output.csv:

    Id,Prediction
    893,178432.51
    ...

Uso:
    python src/predict.py --input data/raw/pipeline_test.csv --output submission.csv

Si el CSV de entrada trae la columna SalePrice, además reporta el RMSE.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from config import ID_COL, MODEL_PT, PIPELINE_TEST_CSV, PRED_COL, ROOT, TARGET
from model import MLP, predict as _predict_one
from utils import load_csv, rmse

_CACHE: dict = {}


def load_artifacts() -> dict:
    """Carga modelo y preprocesadores una sola vez por proceso."""
    if "bundle" in _CACHE:
        return _CACHE["bundle"]

    if not MODEL_PT.exists():
        raise FileNotFoundError(
            f"No existe {MODEL_PT}. Corré primero: python src/03_train_final.py"
        )

    bundle = torch.load(MODEL_PT, weights_only=False)
    models = []
    for m in bundle["members"]:
        net = MLP(m["preprocessor"].n_features, bundle["hparams"])
        net.load_state_dict(m["state_dict"])
        net.eval()
        net.target_mean_ = m["target_mean"]
        net.target_std_ = m["target_std"]
        net.log_target_ = m["log_target"]
        models.append((net, m["preprocessor"]))

    bundle["loaded_models"] = models
    bundle["smearing"] = [m.get("smearing", 1.0) for m in bundle["members"]]
    _CACHE["bundle"] = bundle
    return bundle


def predict_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Predice precios para un DataFrame ya cargado con utils.load_csv.

    Promedia las predicciones de todos los miembros del ensemble. Cada miembro
    transforma los datos con SU propio preprocesador (el que vio al entrenar),
    lo que garantiza que las medias, medianas y categorías sean idénticas a las
    del entrenamiento.
    """
    bundle = load_artifacts()
    X_raw = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])

    preds = []
    for (net, pre), smearing in zip(bundle["loaded_models"], bundle["smearing"]):
        preds.append(_predict_one(net, pre.transform(X_raw)) * smearing)

    # Media geométrica: el modelo se entrenó en escala logarítmica, así que
    # promediar en esa escala es lo consistente con la función de pérdida.
    log_preds = np.log(np.clip(np.array(preds), 1.0, None))
    out = np.exp(log_preds.mean(axis=0))

    # Acotar al rango de precios plausible aprendido del entrenamiento. Una red
    # puede extrapolar muy lejos ante una casa atípica, y en RMSE ese error se
    # paga al cuadrado. El rango es exactamente el mismo que se usó al entrenar.
    floor = bundle.get("price_floor", 1000.0)
    ceiling = bundle.get("price_ceiling", np.inf)
    return np.clip(out, floor, ceiling)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", "-i", default=str(PIPELINE_TEST_CSV),
                    help="CSV de prueba con la misma estructura que pipeline_test.csv")
    ap.add_argument("--output", "-o", default=str(ROOT / "submission.csv"),
                    help="CSV de salida con columnas Id,Prediction")
    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {in_path}")

    df = load_csv(in_path)
    print(f"Entrada: {in_path}  ({df.shape[0]} filas, {df.shape[1]} columnas)")

    preds = predict_dataframe(df)
    ids = df[ID_COL].to_numpy() if ID_COL in df.columns else np.arange(len(df))

    out = pd.DataFrame({ID_COL: ids, PRED_COL: np.round(preds, args.decimals)})
    out.to_csv(args.output, index=False)
    print(f"Salida:  {args.output}  ({len(out)} predicciones)")
    print(out.to_string(index=False))

    if TARGET in df.columns:
        score = rmse(df[TARGET].to_numpy(dtype=float), preds)
        print(f"\nRMSE sobre el archivo de entrada: {score:,.2f}")


if __name__ == "__main__":
    main()
