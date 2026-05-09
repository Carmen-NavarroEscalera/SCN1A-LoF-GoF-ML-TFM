# Clasificación de variantes SCN1A mediante Machine Learning: Diferenciación de mecanismos de pérdida y ganancia de función

**Autor:** Carmen Navarro Escalera  
**Máster en Bioinformática**  
**Trabajo Fin de Máster — 2026**

---

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster sobre la clasificación de variantes missense del gen SCN1A como variantes de pérdida de función (LoF) o ganancia de función (GoF) mediante aprendizaje automático supervisado.

El pipeline integra predictores in silico de última generación obtenidos de dbNSFP y características estructurales del canal Nav1.1 para entrenar y evaluar modelos de clasificación binaria. El modelo definitivo, basado en regresión logística con regularización L2, alcanza un AUC-ROC de 0.885 y una especificidad de 1.000 con el umbral óptimo determinado por el índice de Youden.

---

## Estructura del repositorio

```
SCN1A-LoF-GoF-ML/
├── README.md
├── environment.yml
├── .gitignore
├── scripts/
│   ├── Feature_Importance.py
│   ├── Model_Comparison.py
│   └── Modelo_Final_RL.py
├── figures/
│   ├── figura_votos.png
│   ├── figura_importancia_rfecv.png
│   ├── figura_roc_comparacion.png
│   ├── figura_evaluacion.png
│   ├── figura_shap.png
│   └── figura_curva_aprendizaje.png
├── results/
│   └── tabla_comparacion_modelos.csv
└── data/
    └── README_data.md
```

---

## Datos

Los datos utilizados en este trabajo fueron recopilados manualmente a partir de la literatura científica y bases de datos públicas especializadas, incluyendo ClinVar y publicaciones con caracterización electrofisiológica experimental de variantes SCN1A mediante patch-clamp. Los predictores in silico fueron obtenidos de **dbNSFP** (Database for Nonsynonymous SNPs' Functional Predictions).

El conjunto de datos final incluye **56 variantes missense** de SCN1A con clasificación funcional experimental validada (38 LoF, 18 GoF), publicadas entre 2002 y 2023.

> **Nota:** El archivo de datos (`SCN1A_ML_complete_dbnsfp.xlsx`) no está incluido en este repositorio. Para reproducir los análisis, consulta las instrucciones en `data/README_data.md`.

---

## Scripts

### 1. `Feature_Importance.py`
Selección de características mediante el consenso de tres métodos complementarios basados en XGBoost:
- Eliminación recursiva con validación cruzada (RFECV)
- SelectFromModel
- Importancia por permutación

Genera las figuras de votos por característica e importancia intrínseca, y produce la lista de 9 características seleccionadas por consenso.

### 2. `Model_Comparison.py`
Comparación sistemática de siete clasificadores de aprendizaje supervisado:
- XGBoost, LightGBM, CatBoost
- Random Forest, SVM
- Regresión logística L1 y L2

Evaluación mediante validación cruzada repetida estratificada (k=5, R=10) con siete métricas: AUC-ROC, AUC-PR, F1 macro, MCC, sensibilidad, especificidad, NPV y Brier score. Genera la tabla de comparación y las curvas ROC de todos los modelos.

### 3. `Modelo_Final_RL.py`
Entrenamiento y evaluación del modelo definitivo — regresión logística con regularización L2:
- Optimización de hiperparámetros mediante GridSearchCV exhaustivo (36 combinaciones)
- Evaluación mediante validación cruzada repetida estratificada (k=5, R=10)
- Determinación del umbral óptimo mediante el índice de Youden
- Interpretabilidad mediante valores SHAP con LinearExplainer
- Curva de aprendizaje

---

## Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Carmen-NavarroEscalera/SCN1A-LoF-GoF-ML-TFM.git
cd SCN1A-LoF-GoF-ML-TFM
```

### 2. Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate tfm_env
```

### 4. Ejecutar los scripts en orden

```bash
cd scripts
python Feature_Importance.py
python Model_Comparison.py
python Modelo_Final_RL.py
```
---

## Dependencias principales

| Paquete | Versión |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.3.0 |
| XGBoost | 2.0.3 |
| LightGBM | 4.3.0 |
| CatBoost | 1.2.3 |
| SHAP | 0.44.0 |
| pandas | 2.0.3 |
| numpy | 1.24.3 |
| matplotlib | 3.7.2 |
| seaborn | 0.12.2 |

Para la lista completa de dependencias consulta `environment.yml`.

---

## Resultados principales

| Métrica | Umbral 0.5 | Umbral óptimo (0.732) |
|---|---|---|
| AUC-ROC | 0.885 | — |
| AUC-PR | 0.860 | — |
| Exactitud | 80.4% | 89.3% |
| Sensibilidad (GoF) | 0.778 | 0.667 |
| Especificidad | 0.816 | 1.000 |
| F1 macro | 0.784 | 0.863 |
| Brier score | 0.136 | — |

---

## Cita

Si utilizas este código en tu trabajo, por favor cítalo como:

```
Navarro Escalera C. Clasificación de variantes SCN1A mediante Machine Learning: 
Diferenciación de mecanismos de pérdida y ganancia de función. 
Trabajo Fin de Máster, Máster en Bioinformática, 2026.
```

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.
