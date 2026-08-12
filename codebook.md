# Codebook — Ames Housing

Descripción del dataset, sus variables y las transformaciones aplicadas.

## Fuente y archivos

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `data/raw/train.csv` | 1,168 | 81 | Entrenamiento, incluye `SalePrice` |
| `data/raw/pipeline_test.csv` | 5 | 80 | Muestra del formato del dataset de prueba, sin `SalePrice` |
| `data/raw/expected_output.csv` | 5 | 2 | Formato exigido para la entrega (`Id,Prediction`) |

Cada fila es la venta de una vivienda residencial en Ames, Iowa. La variable objetivo
es `SalePrice`, el precio de venta en dólares.

**Problema:** regresión. **Métrica:** RMSE.

## Variable objetivo

| Estadístico | Valor |
|---|---|
| Media | 181,442 |
| Mediana | 165,000 |
| Desviación estándar | 77,264 |
| Mínimo | 34,900 |
| Máximo | 745,000 |
| Asimetría (skew) | 1.743 |
| Asimetría de `log1p` | 0.121 |

La cola derecha marcada motiva la transformación logarítmica del target.

## Diferencias de formato entre train y test

El archivo de prueba no viene idéntico a `train.csv`. `src/utils.py::load_csv`
normaliza estas diferencias:

| Diferencia | Ejemplo | Tratamiento |
|---|---|---|
| Comillas simples en categóricas | `'Wd Shng'` en `Exterior2nd` | Se eliminan; si no, crearían una categoría no vista |
| Enteros donde train trae float | `LotFrontage`, `MasVnrArea`, `GarageYrBlt` | Conversión numérica explícita |
| Celdas vacías vs. literal `NA` | `Alley`, `PoolQC` | Ambas se leen como nulo |
| Sin salto de línea final | última fila | `pandas` lo maneja; no truncar el archivo |

## Grupos de variables

### Objetivo e identificador
| Variable | Tipo | Descripción |
|---|---|---|
| `Id` | entero | Identificador de la venta. No es predictor; se conserva para la salida |
| `SalePrice` | entero | **Objetivo.** Precio de venta en USD |

### Numéricas continuas (superficies y dimensiones)
| Variable | Rango | Descripción |
|---|---|---|
| `LotFrontage` | 21–313 | Pies lineales de calle conectados al lote |
| `LotArea` | 1,300–215,245 | Superficie del lote (sq ft) |
| `MasVnrArea` | 0–1,600 | Superficie de revestimiento de mampostería |
| `BsmtFinSF1` / `BsmtFinSF2` | 0–5,644 | Superficie terminada del sótano, tipos 1 y 2 |
| `BsmtUnfSF` | 0–2,336 | Superficie sin terminar del sótano |
| `TotalBsmtSF` | 0–6,110 | Superficie total del sótano |
| `1stFlrSF` / `2ndFlrSF` | 334–4,692 | Superficie del primer / segundo nivel |
| `LowQualFinSF` | 0–572 | Superficie terminada de baja calidad |
| `GrLivArea` | 334–5,642 | Superficie habitable sobre el nivel del suelo |
| `GarageArea` | 0–1,418 | Superficie del garaje |
| `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch` | 0–870 | Superficies exteriores |
| `PoolArea` | 0–738 | Superficie de la piscina |
| `MiscVal` | 0–15,500 | Valor de elementos misceláneos |

### Numéricas discretas (conteos)
`BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`,
`KitchenAbvGr`, `TotRmsAbvGrd`, `Fireplaces`, `GarageCars`.

### Temporales
| Variable | Rango | Descripción |
|---|---|---|
| `YearBuilt` | 1872–2010 | Año de construcción |
| `YearRemodAdd` | 1950–2010 | Año de remodelación (= `YearBuilt` si no hubo) |
| `GarageYrBlt` | 1900–2010 | Año de construcción del garaje |
| `MoSold` | 1–12 | Mes de venta. Tratada como **categórica**: no tiene orden lineal |
| `YrSold` | 2006–2010 | Año de venta |

### Ordinales — calidad y condición

Escala general (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`,
`KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`):

| Código | Valor | Significado |
|---|---|---|
| `None` | 0 | No aplica / no tiene |
| `Po` | 1 | Pobre |
| `Fa` | 2 | Regular |
| `TA` | 3 | Típico / promedio |
| `Gd` | 4 | Bueno |
| `Ex` | 5 | Excelente |

Otras ordinales con su mapeo:

| Variable | Mapeo |
|---|---|
| `OverallQual` / `OverallCond` | 1–10, ya numéricas |
| `BsmtExposure` | None=0, No=1, Mn=2, Av=3, Gd=4 |
| `BsmtFinType1` / `BsmtFinType2` | None=0, Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6 |
| `GarageFinish` | None=0, Unf=1, RFn=2, Fin=3 |
| `Functional` | Sal=0, Sev=1, Maj2=2, Maj1=3, Mod=4, Min2=5, Min1=6, Typ=7 |
| `CentralAir` | N=0, Y=1 |
| `PavedDrive` | N=0, P=1, Y=2 |
| `LotShape` | IR3=0, IR2=1, IR1=2, Reg=3 |
| `LandSlope` | Sev=0, Mod=1, Gtl=2 |
| `Street` | Grvl=0, Pave=1 |
| `Alley` | None=0, Grvl=1, Pave=2 |
| `Utilities` | ELO=0, NoSeWa=1, NoSewr=2, AllPub=3 |
| `Fence` | None=0, MnWw=1, GdWo=2, MnPrv=3, GdPrv=4 |

### Nominales (one-hot)

`MSSubClass` (tratada como categórica pese a ser numérica), `MSZoning`,
`LandContour`, `LotConfig`, `Neighborhood` (25 valores), `Condition1`, `Condition2`,
`BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`,
`MasVnrType`, `Foundation`, `Heating`, `Electrical`, `GarageType`, `MiscFeature`,
`SaleType`, `SaleCondition`.

## Valores nulos y su tratamiento

19 columnas tienen nulos. **La mayoría no son datos faltantes: son ausencia del
elemento.**

| Variable | % nulos | Significado del nulo | Imputación |
|---|---|---|---|
| `PoolQC` | 99.5 | Sin piscina | `"None"` |
| `MiscFeature` | 96.1 | Sin elementos misceláneos | `"None"` |
| `Alley` | 93.7 | Sin acceso por callejón | `"None"` |
| `Fence` | 80.1 | Sin cerca | `"None"` |
| `MasVnrType` | 58.5 | Sin revestimiento | `"None"` |
| `FireplaceQu` | 46.8 | Sin chimenea | `"None"` |
| `LotFrontage` | 18.6 | **Dato faltante real** | Mediana del vecindario |
| `GarageYrBlt` | 5.5 | Sin garaje | `YearBuilt` de la casa |
| `GarageCond`, `GarageType`, `GarageFinish`, `GarageQual` | 5.5 | Sin garaje | `"None"` |
| `BsmtQual`, `BsmtCond`, `BsmtFinType1`, `BsmtFinType2`, `BsmtExposure` | 2.4 | Sin sótano | `"None"` |
| `MasVnrArea` | 0.5 | Sin revestimiento | `0` |
| `Electrical` | 0.1 | Dato faltante real | Moda |

## Features derivadas

Generadas en `src/preprocessing.py::add_engineered_features`. Todas son
combinaciones deterministas de columnas existentes.

| Feature | Fórmula | Correlación con `SalePrice` |
|---|---|---|
| `QualxSF` | `OverallQual × TotalSF` | 0.837 |
| `QualxGrLiv` | `OverallQual × GrLivArea` | 0.817 |
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 0.766 |
| `TotalBath` | `FullBath + 0.5·HalfBath + BsmtFullBath + 0.5·BsmtHalfBath` | 0.621 |
| `OverallScore` | `OverallQual × OverallCond` | 0.555 |
| `HasFireplace` | `Fireplaces > 0` | 0.471 |
| `TotalPorchSF` | Suma de las 5 superficies exteriores | 0.389 |
| `HasGarage` | `GarageArea > 0` | 0.248 |
| `IsNew` | `YrSold = YearBuilt` | 0.213 |
| `HasBsmt` | `TotalBsmtSF > 0` | 0.156 |
| `Has2ndFloor` | `2ndFlrSF > 0` | 0.130 |
| `HasPool` | `PoolArea > 0` | 0.117 |
| `IsRemodeled` | `YearRemodAdd ≠ YearBuilt` | −0.029 |
| `RemodAge` | `YrSold − YearRemodAdd` | −0.510 |
| `HouseAge` | `YrSold − YearBuilt` | −0.516 |

`QualxSF` (0.837) y `QualxGrLiv` (0.817) superan en correlación a cualquier variable
original — la mayor es `OverallQual` con 0.786.

## Transformaciones del pipeline

Aplicadas en orden, todas ajustadas **solo** con datos de entrenamiento:

1. Normalización de formato del CSV (`load_csv`).
2. Imputación de nulos por significado.
3. Mapeo de ordinales a enteros.
4. Generación de las 15 features derivadas.
5. `log1p` sobre las numéricas con `|skew| > 0.75` (46 columnas).
6. One-hot de las nominales, con categorías fijadas en entrenamiento.
7. Estandarización z-score.

**Matriz resultante: 260 columnas.**

## Casos atípicos documentados

| `Id` | `GrLivArea` | `SalePrice` | `OverallQual` | `SaleCondition` |
|---|---|---|---|---|
| 524 | 4,676 | 184,750 | 10 | Partial |
| 1299 | 5,642 | 160,000 | 10 | Partial |

Son casas de máxima calidad vendidas **sin terminar**, así que su precio no refleja el
inmueble completo. Se remueven del conjunto de entrenamiento (decisión validada
empíricamente en `src/02c_robustez.py`), nunca de validación ni del holdout.
