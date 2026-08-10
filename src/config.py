"""Rutas, semillas y contratos de columnas del proyecto.

Todo el resto del pipeline importa de aquí: ningún script debe tener rutas
ni listas de columnas hardcodeadas.
"""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TRAIN_CSV = DATA_RAW / "train.csv"
PIPELINE_TEST_CSV = DATA_RAW / "pipeline_test.csv"
EXPECTED_OUTPUT_CSV = DATA_RAW / "expected_output.csv"

PREPROCESSOR_PKL = MODELS / "preprocessor.pkl"
MODEL_PT = MODELS / "mlp.pt"
EXPERIMENTS_CSV = REPORTS / "experiments.csv"

for _d in (DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibilidad ------------------------------------------------------
SEED = 42

# --- Contrato de columnas --------------------------------------------------
ID_COL = "Id"
TARGET = "SalePrice"
PRED_COL = "Prediction"

# Categóricas cuyo NaN significa "la casa no tiene ese elemento", no "dato
# faltante". Se imputan con el literal "None" antes de codificar.
NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Numéricas cuyo NaN significa "no tiene" -> 0.
NA_MEANS_ZERO = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# Variables ordinales: categorías con orden natural. Mapear a enteros preserva
# la relación monótona con el precio y usa 1 columna en vez de k dummies.
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS = {
    "ExterQual": QUALITY_MAP,
    "ExterCond": QUALITY_MAP,
    "BsmtQual": QUALITY_MAP,
    "BsmtCond": QUALITY_MAP,
    "HeatingQC": QUALITY_MAP,
    "KitchenQual": QUALITY_MAP,
    "FireplaceQu": QUALITY_MAP,
    "GarageQual": QUALITY_MAP,
    "GarageCond": QUALITY_MAP,
    "PoolQC": QUALITY_MAP,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                   "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Street": {"Grvl": 0, "Pave": 1},
    "Alley": {"None": 0, "Grvl": 1, "Pave": 2},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}

# Numérica por tipo pero categórica por significado: el código de tipo de
# vivienda no tiene orden (20 no es "menor" que 60).
FORCE_CATEGORICAL = ["MSSubClass", "MoSold"]
