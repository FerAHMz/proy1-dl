"""Etapa 1: el análisis exploratorio.

Saca las tablas y figuras que uso en la sección 2.1 del escrito. Escribe en
reports/figures/*.png y reports/eda_*.csv, y no toca nada de data/raw.

Uso:  python src/01_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES, REPORTS, TARGET, TRAIN_CSV
from utils import load_csv, set_seed

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120


def main() -> None:
    set_seed()
    df = load_csv(TRAIN_CSV)
    y = df[TARGET]

    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    # --- Tipos de variables -------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(
        ["Id", TARGET], errors="ignore"
    )
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    print(f"Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    tipos = pd.DataFrame(
        {
            "variable": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
            "n_nulos": df.isna().sum().values,
            "pct_nulos": (df.isna().mean() * 100).round(2).values,
        }
    ).sort_values("pct_nulos", ascending=False)
    tipos.to_csv(REPORTS / "eda_variables.csv", index=False)

    # --- Descriptivas -------------------------------------------------------
    desc = df[num_cols.tolist() + [TARGET]].describe().T
    desc["skew"] = df[num_cols.tolist() + [TARGET]].skew()
    desc["kurtosis"] = df[num_cols.tolist() + [TARGET]].kurtosis()
    desc.round(3).to_csv(REPORTS / "eda_descriptivas.csv")

    print("\nTarget SalePrice:")
    print(
        f"  media={y.mean():,.0f}  mediana={y.median():,.0f}  std={y.std():,.0f}"
        f"  min={y.min():,.0f}  max={y.max():,.0f}  skew={y.skew():.3f}"
    )

    # --- Nulos --------------------------------------------------------------
    nulos = tipos[tipos.n_nulos > 0]
    print(f"\nColumnas con nulos: {len(nulos)}")
    print(nulos[["variable", "n_nulos", "pct_nulos"]].head(20).to_string(index=False))

    if len(nulos):
        fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(nulos))))
        sns.barplot(data=nulos, y="variable", x="pct_nulos", color="#4C72B0", ax=ax)
        ax.set_title("Porcentaje de valores nulos por variable")
        ax.set_xlabel("% nulos")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / "01_nulos.png")
        plt.close(fig)

    # --- Distribución del target -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(y, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"SalePrice (skew={y.skew():.2f})")
    sns.histplot(np.log1p(y), kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title(f"log1p(SalePrice) (skew={np.log1p(y).skew():.2f})")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_target_distribucion.png")
    plt.close(fig)

    # --- Correlaciones ------------------------------------------------------
    corr = df[num_cols.tolist() + [TARGET]].corr(numeric_only=True)[TARGET]
    corr = corr.drop(TARGET).sort_values(ascending=False)
    corr.round(4).to_csv(REPORTS / "eda_correlaciones.csv", header=["corr_SalePrice"])
    print("\nTop 12 correlaciones con SalePrice:")
    print(corr.head(12).round(3).to_string())
    print("\nMás negativas:")
    print(corr.tail(5).round(3).to_string())

    top = corr.abs().sort_values(ascending=False).head(12).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        df[top + [TARGET]].corr(numeric_only=True),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        square=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title("Correlación entre las 12 features más asociadas al precio")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_correlaciones.png")
    plt.close(fig)

    # --- Relación feature vs target ----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), corr.head(6).index):
        sns.scatterplot(x=df[col], y=y, alpha=0.45, s=18, ax=ax, color="#4C72B0")
        ax.set_title(f"{col} (r={corr[col]:.2f})")
        ax.set_ylabel("SalePrice")
    fig.suptitle("Features numéricas más correlacionadas vs. SalePrice")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_features_vs_target.png")
    plt.close(fig)

    # --- Categóricas relevantes --------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, col in zip(axes.ravel(), ["Neighborhood", "ExterQual", "KitchenQual", "GarageType"]):
        order = df.groupby(col, observed=True)[TARGET].median().sort_values().index
        sns.boxplot(data=df, x=col, y=TARGET, order=order, ax=ax, fliersize=2)
        ax.set_title(f"SalePrice por {col}")
        ax.tick_params(axis="x", rotation=90 if col == "Neighborhood" else 0)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_categoricas.png")
    plt.close(fig)

    # --- Outliers -----------------------------------------------------------
    # Los casos ya conocidos de Ames: casas enormes vendidas baratas porque
    # fueron ventas parciales. Acá solo los reporto; qué hago con ellos lo
    # decido más adelante.
    out = df[(df["GrLivArea"] > 4000) & (df[TARGET] < 300000)]
    print(f"\nOutliers GrLivArea>4000 & SalePrice<300k: {len(out)}")
    if len(out):
        print(out[["Id", "GrLivArea", TARGET, "SaleCondition", "OverallQual"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=df["GrLivArea"], y=y, alpha=0.5, s=22, ax=ax)
    if len(out):
        sns.scatterplot(x=out["GrLivArea"], y=out[TARGET], color="crimson",
                        s=90, marker="X", ax=ax, label="outlier candidato")
    ax.set_title("GrLivArea vs SalePrice — candidatos a outlier")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_outliers.png")
    plt.close(fig)

    # IQR por variable, para tener una referencia numérica y no solo la gráfica.
    q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
    iqr = q3 - q1
    n_out = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).sum()
    n_out.sort_values(ascending=False).to_csv(REPORTS / "eda_outliers_iqr.csv",
                                              header=["n_outliers_iqr"])

    print(f"\nFiguras en {FIGURES}")
    print(f"Tablas en {REPORTS}")


if __name__ == "__main__":
    main()
