# Proyecto 1 — Competencia de modelación

**CC3092 Deep Learning y sistemas inteligentes** · Universidad del Valle de Guatemala

Multi-Layer Perceptron para predecir el precio de venta de viviendas (dataset Ames
Housing). Métrica objetivo: **RMSE**.

## Resultado de la competencia (17 de agosto)

RMSE del mejor envío en el dataset de prueba real (292 casas): **24,833 USD**,
por encima del baseline de regresión lineal del curso (29,474).

El modelo de competencia (`src/04_competencia.py`) es un ensemble de 40 MLPs:
4 arquitecturas distintas (256-128-64 GELU, 512-256-128, 128-64 y 256-128-64
SiLU) x 10 folds, entrenado con todo `train.csv`. Durante la ventana de envíos
probé de forma sistemática: acotar/no acotar el rango de predicción (quitarlo
mejoró 4,357), reentrenar con todos los datos, pseudo-labeling y target
encoding de vecindario. Los últimos dos mejoraron la validación local pero no
el test — evidencia de que el RMSE del test lo dominan unos pocos casos
extremos, consistente con lo observado en el holdout.

## Resultados del desarrollo (holdout propio)

| Métrica | Valor |
|---|---|
| RMSE en holdout (176 casas nunca vistas) | **28,091 USD** |
| RMSE en validación cruzada (out-of-fold) | 26,473 USD |
| Error porcentual absoluto medio (MAPE) | 8.7% |
| Predicciones dentro del ±10% del precio real | 74% |
| Predicciones dentro del ±20% del precio real | 91% |

La coincidencia entre validación cruzada y holdout (6% de diferencia) indica que el
proceso de selección de modelo no sobreajustó al protocolo de validación.

**Modelo final:** ensemble de 15 MLPs (5 folds × 3 semillas), arquitectura
`260 → 256 → 128 → 64 → 1` con BatchNorm, GELU y Dropout 0.2.

## Estructura

```
proy1-dl/
├── data/raw/                  datos originales (inmutables)
│   ├── train.csv              1,168 filas x 81 columnas
│   ├── pipeline_test.csv      muestra del formato del dataset de prueba
│   ├── test_features.csv      dataset de prueba real (entregado el 17 de agosto)
│   └── expected_output.csv    formato exigido para la entrega
├── notebooks/
│   ├── 01_eda.ipynb                      análisis exploratorio
│   ├── 02_metodologia_iteraciones.ipynb  protocolo y 16 iteraciones
│   ├── 03_modelo_final_discusion.ipynb   modelo final, diagnóstico, residuos
│   └── 04_prediccion.ipynb               predicción sobre un dataset nuevo
├── src/
│   ├── config.py              rutas, semillas y contratos de columnas
│   ├── utils.py               carga robusta de CSVs, semillas, métricas
│   ├── preprocessing.py       pipeline de preprocesamiento serializable
│   ├── model.py               definición del MLP y bucle de entrenamiento
│   ├── experiment.py          protocolo de evaluación (holdout + K-fold)
│   ├── 01_eda.py              genera tablas y figuras del EDA
│   ├── 02_experiments.py      historial de 16 iteraciones
│   ├── 02b_refinamiento.py    desempate con CV repetida
│   ├── 02c_robustez.py        corrección de early stopping, outliers y techo
│   ├── 02d_calibracion.py     calibración del techo y smearing
│   ├── 03_train_final.py      entrena el modelo final y evalúa el holdout
│   ├── 04_competencia.py      modelo de la competencia (ensemble diverso)
│   ├── predict.py             predicción sobre un dataset nuevo
│   └── run_pipeline.py        orquestador de todas las etapas
├── models/                    artefactos entrenados (mlp.pt, preprocessor.pkl)
├── reports/                   tablas de experimentos y métricas
│   └── figures/               figuras generadas
├── codebook.md                descripción del dataset y sus variables
├── submission_competencia.csv mejor envío de la competencia (RMSE 24,833)
└── requirements.txt
```

## Entorno

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.11. Corre en CPU; no requiere GPU.

## Uso

### Predecir sobre un dataset de prueba

Es el caso principal. No requiere reentrenar: los artefactos ya están en `models/`.

```bash
python src/predict.py --input data/raw/pipeline_test.csv --output submission.csv
```

Genera un CSV con el formato exacto de `expected_output.csv`:

```
Id,Prediction
893,148737.44
1106,324601.12
...
```

Si el archivo de entrada incluye la columna `SalePrice`, además reporta el RMSE.

La versión en notebook del mismo procedimiento está en `notebooks/04_prediccion.ipynb`.

### Reproducir todo desde cero

```bash
python src/run_pipeline.py              # todas las etapas (~25 min en CPU)
python src/run_pipeline.py --skip-exp   # solo EDA y modelo final (~4 min)
```

O etapa por etapa:

```bash
python src/01_eda.py            # tablas y figuras del EDA
python src/02_experiments.py    # 16 iteraciones -> reports/experiments.csv
python src/02b_refinamiento.py  # CV repetida    -> reports/refinamiento.csv
python src/02c_robustez.py      # correcciones   -> reports/robustez.csv
python src/02d_calibracion.py   # calibración    -> reports/calibracion.csv
python src/03_train_final.py    # modelo final   -> models/
```

Todas las etapas fijan `SEED = 42` en numpy, torch y random.

### Notebooks

```bash
jupyter notebook
```

Se ejecutan en orden y esperan ser abiertos desde `notebooks/` (agregan `../src` al
path). Los outputs vienen incrustados.

## Metodología

### Protocolo de evaluación

```
train.csv (1,168 filas)
  ├── HOLDOUT     15%  (176 filas)  ── apartado, intacto hasta el final
  └── DESARROLLO  85%  (992 filas)
        └── 5-fold CV  ← todas las decisiones se toman aquí
```

Dos reglas que sostienen la validez del resultado:

1. **El holdout se toca una sola vez**, al final. Si se usara para elegir
   hiperparámetros dejaría de ser una estimación imparcial.
2. **El preprocesador se ajusta dentro de cada fold.** Calcular medianas, categorías y
   parámetros de normalización sobre todo el dataset antes de partirlo filtraría
   información del fold de validación (*data leakage*).

Los splits son estratificados por decil de precio.

### Preprocesamiento

- Imputación de nulos **según su significado**: en este dataset la mayoría de los
  nulos indican ausencia del elemento (`PoolQC` nulo = sin piscina), no dato faltante.
- Codificación mixta: ordinal para las calidades (`Po` < `Fa` < `TA` < `Gd` < `Ex`),
  one-hot para las nominales.
- 15 features derivadas: superficies totales, edad de la casa, interacciones
  calidad × área.
- `log1p` sobre las numéricas sesgadas y estandarización z-score.

Resultado: **260 features**. Detalle completo en [`codebook.md`](codebook.md).

### Control de sobreajuste

| Mecanismo | Qué evita |
|---|---|
| Holdout apartado desde el inicio | Sobreajuste por selección de modelo |
| 5-fold CV para todas las decisiones | Elegir por suerte de un solo split |
| CV repetida (3 semillas) para desempatar | Confundir ruido con mejora real |
| Early stopping en escala logarítmica | Sobreajuste por exceso de épocas, con señal estable |
| Dropout 0.2 + BatchNorm + weight decay | Memorización de la red |
| Ensemble de 15 miembros | Varianza por inicialización y split |
| Techo de predicción calibrado por CV | Extrapolación fuera del rango observado |

## Hallazgos principales

**El problema era sobreajuste, no falta de capacidad.** El baseline alcanzaba un RMSE
de entrenamiento de 4,835 contra 42,718 en validación. Hacer la red más profunda sin
regularizar solo mejoró la memorización.

**El mayor factor no fue la arquitectura sino acotar el rango de predicción**
(−6,171 USD). La red llegaba a predecir 802,302 USD cuando el máximo observado es
745,000, y en RMSE ese error se paga al cuadrado.

**La métrica del early stopping importaba tanto como los hiperparámetros.**
Seleccionar el checkpoint por RMSE en escala original resultó ser una señal
dominada por una o dos casas extremas por fold: hubo miembros que se detuvieron en la
época 2. Cambiar a escala logarítmica redujo la desviación entre semillas de ±3,318 a
±766.

**El ruido del protocolo superaba las diferencias que se querían medir.** La
desviación entre folds (9,000–17,000 USD) era varias veces mayor que las diferencias
entre las mejores configuraciones (1,000–2,000 USD), lo que obligó a usar validación
cruzada repetida para decidir.

**La competencia dejó dos lecciones más.** Quitar el techo de predicción mejoró
4,357 USD en el test real — el techo estaba calibrado contra un holdout que contenía
una venta parcial patológica que el test no tenía. Y tanto el pseudo-labeling como el
target encoding mejoraron la validación local sin mover el test: cuando la métrica la
dominan pocos casos extremos, la señal local deja de predecir el resultado.

**Dos hipótesis resultaron falsas y quedaron documentadas:** conservar los outliers de
venta parcial para que el modelo aprendiera el patrón (empeoró), y la corrección de
smearing de Duan (ganancia dentro del ruido, porque la pérdida Huber en escala
logarítmica ya centra bien los residuos).

## Limitaciones

- **El tamaño del dataset es el techo real**: 1,168 filas para 260 features, ~4.5
  observaciones por feature. Por eso ensanchar la red no ayudó.
- **El RMSE absoluto es una métrica frágil aquí.** Un solo caso del holdout (`Id 524`,
  una venta parcial cuyo precio depende de información ausente del dataset) aporta el
  47% del error cuadrático total. Sin él, el RMSE sería 20,486.
- **Un MLP no es el modelo natural para datos tabulares**; los métodos de árboles con
  boosting suelen superarlo. El proyecto exige un MLP.
- Si el dataset de prueba trae categorías no vistas en entrenamiento, el one-hot las
  codifica como todo-ceros: degradación controlada, pero degradación.
