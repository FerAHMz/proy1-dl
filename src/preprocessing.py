"""Preprocesamiento. Lo ajusto una vez con train y lo vuelvo a aplicar igual
al dataset de prueba, que es lo que pide el enunciado.

Lo guardo completo en ``models/preprocessor.pkl`` para que el día de la
presentación no tenga que recalcular medias ni categorías: solo lo cargo.
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
    """Features que armé combinando columnas que ya existen. Como son puras
    operaciones entre ellas, no se me cuela información del target ni dependen
    de cómo parta los datos.
    """
    X = X.copy()

    # La red podría deducir la superficie total sola, pero dándosela directa
    # converge más rápido, y resultó ser lo que más correlaciona con el precio.
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

    # Prefiero la edad al momento de la venta que los años absolutos: si no,
    # la red tiene que hacer la resta por su cuenta.
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]
    X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    X["IsNew"] = (X["YrSold"] == X["YearBuilt"]).astype(int)

    # Banderas de "tiene o no tiene", para no confundirlo con "tiene poco".
    X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)
    X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
    X["HasBsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
    X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)
    X["Has2ndFloor"] = (X["2ndFlrSF"].fillna(0) > 0).astype(int)

    # Calidad por superficie: el metro cuadrado no vale lo mismo en una casa de
    # calidad 3 que en una de calidad 9.
    X["QualxSF"] = X["OverallQual"] * X["TotalSF"]
    X["QualxGrLiv"] = X["OverallQual"] * X["GrLivArea"]
    X["OverallScore"] = X["OverallQual"] * X["OverallCond"]

    return X


@dataclass
class Preprocessor:
    """Aprende de train y transforma cualquier split con esos mismos números.

    Lo que saco en ``fit`` y reuso tal cual en ``transform``: medianas para
    imputar, categorías vistas, medias y desviaciones para estandarizar.
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

        # Estos NaN significan "no tiene", así que van a "None" o 0 en vez de a la
        # mediana.
        for col in NA_MEANS_NONE:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        for col in NA_MEANS_ZERO:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # Si no hay garaje uso el año de la casa; poner 0 me rompía la escala.
        if "GarageYrBlt" in X.columns:
            X["GarageYrBlt"] = pd.to_numeric(
                X["GarageYrBlt"], errors="coerce"
            ).fillna(X["YearBuilt"])

        # Ordinales a enteros, respetando su orden.
        for col, mapping in ORDINAL_MAPS.items():
            if col in X.columns:
                X[col] = (
                    X[col].fillna("None").map(mapping).astype(float)
                )
                # Si aparece algo que no está en el mapa, le pongo el valor de
                # en medio para no sesgarlo hacia ningún extremo.
                X[col] = X[col].fillna(float(np.median(list(mapping.values()))))

        return X

    def _impute_lotfrontage(self, X: pd.DataFrame) -> pd.DataFrame:
        """LotFrontage es el nulo que más aparece (~18%). Como el frente del
        lote depende de la traza del vecindario, lo lleno con la mediana de su
        vecindario, calculada en train y no en el split que esté procesando."""
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

        # Me guardo las categorías que vi en train. Si en el test aparece una
        # nueva, su fila queda en ceros en vez de abrir una columna que la red
        # nunca vio.
        self.categories_ = {
            c: sorted(X[c].fillna("None").astype(str).unique().tolist())
            for c in self.categorical_cols_
        }

        # log1p a las numéricas muy sesgadas: aplasta las colas largas de
        # LotArea y las superficies, y el entrenamiento va más estable.
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
        """Arma la matriz final siempre con las columnas en el mismo orden."""
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
            # Hago el one-hot a mano para que salgan las mismas columnas y en
            # el mismo orden que en fit.
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

        # Relleno las columnas que fit vio y aquí puedan faltar.
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
