"""Pipeline de preprocesamiento: se ajusta UNA vez con train y se reaplica
idéntico al dataset de prueba (requisito 2 de las recomendaciones del PDF).

Se serializa completo en ``models/preprocessor.pkl``, así que el día de la
presentación no hay que recalcular ni medias ni categorías: se cargan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    FORCE_CATEGORICAL,
    NA_MEANS_NONE,
    NA_MEANS_ZERO,
    ORDINAL_MAPS,
)


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas. Todas son combinaciones deterministas de columnas
    existentes, así que no filtran información del target ni dependen del split.
    """
    X = X.copy()

    # Superficie total habitable: el MLP puede aprenderla, pero dársela explícita
    # acelera la convergencia y es la variable más correlacionada con el precio.
    X["TotalSF"] = (
        X["TotalBsmtSF"].fillna(0) + X["1stFlrSF"].fillna(0) + X["2ndFlrSF"].fillna(0)
    )
    X["TotalBath"] = (
        X["FullBath"].fillna(0)
        + 0.5 * X["HalfBath"].fillna(0)
        + X["BsmtFullBath"].fillna(0)
        + 0.5 * X["BsmtHalfBath"].fillna(0)
    )
    X["TotalPorchSF"] = (
        X["OpenPorchSF"].fillna(0)
        + X["EnclosedPorch"].fillna(0)
        + X["3SsnPorch"].fillna(0)
        + X["ScreenPorch"].fillna(0)
        + X["WoodDeckSF"].fillna(0)
    )

    # Edad al momento de la venta y años desde la remodelación: más informativas
    # que los años absolutos, que el modelo tendría que restar por su cuenta.
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
    X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    X["IsNew"] = (X["YrSold"] == X["YearBuilt"]).astype(int)

    # Indicadores de presencia: separan el "no tiene" del "tiene poco".
    X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)
    X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
    X["HasBsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
    X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)
    X["Has2ndFloor"] = (X["2ndFlrSF"].fillna(0) > 0).astype(int)

    # Interacciones calidad x superficie: el precio no escala igual por m2 en una
    # casa de calidad 3 que en una de calidad 9.
    X["QualxSF"] = X["OverallQual"] * X["TotalSF"]
    X["QualxGrLiv"] = X["OverallQual"] * X["GrLivArea"]
    X["OverallScore"] = X["OverallQual"] * X["OverallCond"]

    return X


@dataclass
class Preprocessor:
    """Ajusta con train, transforma cualquier split con los MISMOS parámetros.

    Parámetros aprendidos en ``fit`` y reutilizados tal cual en ``transform``:
    medianas de imputación, categorías vistas, medias y desviaciones estándar.
    """

    skew_threshold: float = 0.75

    feature_names_: list[str] = field(default_factory=list)
    numeric_cols_: list[str] = field(default_factory=list)
    categorical_cols_: list[str] = field(default_factory=list)
    medians_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    skewed_cols_: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    lotfrontage_by_neighborhood_: dict = field(default_factory=dict)
    lotfrontage_global_: float = 0.0

    # --- helpers -----------------------------------------------------------
    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in FORCE_CATEGORICAL:
            if col in X.columns:
                X[col] = X[col].astype("Int64").astype(str)

        # NaN con significado semántico -> "None" / 0, no imputación estadística.
        for col in NA_MEANS_NONE:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in NA_MEANS_ZERO:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # GarageYrBlt sin garaje: usar el año de construcción de la casa evita
        # inventar un año 0 que distorsiona la escala.
        if "GarageYrBlt" in X.columns:
            X["GarageYrBlt"] = pd.to_numeric(
                X["GarageYrBlt"], errors="coerce"
            ).fillna(X["YearBuilt"])

        # Ordinales -> enteros con su orden natural.
        for col, mapping in ORDINAL_MAPS.items():
            if col in X.columns:
                X[col] = (
                    X[col].fillna("None").map(mapping).astype(float)
                )
                # Categoría no vista en el mapa -> mediana del mapa (valor neutro).
                X[col] = X[col].fillna(float(np.median(list(mapping.values()))))

        return X

    def _impute_lotfrontage(self, X: pd.DataFrame) -> pd.DataFrame:
        """LotFrontage es el nulo más frecuente (~18%). El frente del lote está
        determinado por el vecindario, así que se imputa con la mediana del
        vecindario aprendida en train (no del split actual)."""
        X = X.copy()
        if "LotFrontage" not in X.columns:
            return X
        lf = pd.to_numeric(X["LotFrontage"], errors="coerce")
        fill = X["Neighborhood"].map(self.lotfrontage_by_neighborhood_)
        X["LotFrontage"] = lf.fillna(fill).fillna(self.lotfrontage_global_)
        return X

    # --- API ---------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        X = self._basic_clean(X)

        lf = pd.to_numeric(X["LotFrontage"], errors="coerce")
        self.lotfrontage_by_neighborhood_ = (
            lf.groupby(X["Neighborhood"]).median().to_dict()
        )
        self.lotfrontage_global_ = float(lf.median())
        X = self._impute_lotfrontage(X)

        X = add_engineered_features(X)

        self.numeric_cols_ = sorted(
            X.select_dtypes(include=[np.number]).columns.tolist()
        )
        self.categorical_cols_ = sorted(
            [c for c in X.columns if c not in self.numeric_cols_]
        )

        self.medians_ = {
            c: float(pd.to_numeric(X[c], errors="coerce").median())
            for c in self.numeric_cols_
        }

        # Categorías vistas en train: en transform, cualquier valor nuevo cae en
        # "__other__" en vez de crear una columna que el modelo nunca vio.
        self.categories_ = {
            c: sorted(X[c].fillna("None").astype(str).unique().tolist())
            for c in self.categorical_cols_
        }

        # log1p sobre las numéricas muy sesgadas: comprime colas largas
        # (LotArea, superficies) y estabiliza el entrenamiento.
        num = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        skews = num.fillna(pd.Series(self.medians_)).skew()
        self.skewed_cols_ = [
            c for c in self.numeric_cols_
            if abs(skews.get(c, 0)) > self.skew_threshold and (num[c].min() >= 0)
        ]

        M = self._to_matrix(X)
        self.mean_ = M.mean(axis=0)
        self.std_ = M.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # columnas constantes: no dividir por ~0
        return self

    def _to_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Construye la matriz de diseño en un orden de columnas fijo."""
        parts, names = [], []

        num = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(pd.Series(self.medians_))
        for c in self.skewed_cols_:
            num[c] = np.log1p(num[c].clip(lower=0))
        parts.append(num.to_numpy(dtype=float))
        names.extend(self.numeric_cols_)

        for c in self.categorical_cols_:
            cats = self.categories_[c]
            vals = X[c].fillna("None").astype(str)
            # One-hot manual: garantiza mismas columnas y mismo orden que en fit.
            block = np.zeros((len(X), len(cats)), dtype=float)
            idx = pd.Series(vals).map({k: i for i, k in enumerate(cats)})
            known = idx.notna().to_numpy()
            block[np.arange(len(X))[known], idx[known].astype(int)] = 1.0
            parts.append(block)
            names.extend([f"{c}={k}" for k in cats])

        self.feature_names_ = names
        return np.hstack(parts)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._basic_clean(X)
        X = self._impute_lotfrontage(X)
        X = add_engineered_features(X)

        # Asegurar que existan todas las columnas que vio fit, en su orden.
        for c in self.numeric_cols_:
            if c not in X.columns:
                X[c] = self.medians_[c]
        for c in self.categorical_cols_:
            if c not in X.columns:
                X[c] = "None"

        M = self._to_matrix(X)
        M = (M - self.mean_) / self.std_
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
