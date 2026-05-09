"""
Comparación de modelos de Machine Learning para clasificar variantes LoF vs GoF en SCN1A.

Incluye:
- Preprocesado de variables numéricas y categóricas
- Balanceo mediante class_weight y scale_pos_weight
- Evaluación con RepeatedStratifiedKFold (5x10)
- Métricas: AUC-ROC, F1 macro, MCC, Sensibilidad, Especificidad, NPV, Brier score
- Modelos: XGBoost, LightGBM, CatBoost, Random Forest, SVM, Regresión logística L1 y L2
- Curvas ROC comparativas
- Exportación de tabla de resultados y figuras

Requiere:
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (make_scorer, matthews_corrcoef, f1_score,
                             brier_score_loss, confusion_matrix,
                             roc_curve, auc)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================
# 1. Rutas robustas
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "..", "data", "SCN1A_ML_complete_dbnsfp.xlsx")
FIG_DIR = os.path.join(BASE_DIR, "..", "figures")
RES_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ============================
# 2. Cargar datos
# ============================
df = pd.read_excel(DATA_FILE)

# ============================
# 3. Pre-procesado
# ============================
def clean_text_column(col):
    return (
        col.astype(str)
           .str.strip().str.lower()
           .str.replace("_", " ", regex=False)
           .str.replace("-", " ", regex=False)
           .str.replace(r"\s+", " ", regex=True)
    )

for col in ["Dominio", "Alelo_Referencia", "Alelo_Alternativo", "Clase_Funcional"]:
    df[col] = clean_text_column(df[col])

def agrupar_protein_domain(x):
    x = str(x).lower()
    if "diii div linker" in x: return "diii_div_linker"
    elif x.startswith("di "):   return "di"
    elif x.startswith("dii "):  return "dii"
    elif x.startswith("diii "): return "diii"
    elif x.startswith("div "):  return "div"
    elif "c terminal" in x:     return "c_terminal"
    else:                        return "otros"

df["Dominio_agrupado"] = df["Dominio"].apply(agrupar_protein_domain)
df["Clase_Funcional"] = df["Clase_Funcional"].map({"lof": 0, "gof": 1})
df = df.dropna(subset=["Clase_Funcional"])

y = df["Clase_Funcional"].astype(int)

n_lof = (y == 0).sum()
n_gof = (y == 1).sum()
scale_pos_weight = n_lof / n_gof
print(f"Distribución — LoF: {n_lof}, GoF: {n_gof}, ratio: {scale_pos_weight:.2f}")

# ============================
# 4. Features seleccionadas por consenso
# ============================
selected_numeric = [
    "ESM1b_converted_rankscore", "PrimateAI_rankscore",
    "SIFT4G_converted_rankscore", "Posicion"
]
selected_categorical = ["Dominio_agrupado", "Alelo_Referencia", "Alelo_Alternativo"]

X = df[selected_numeric + selected_categorical].copy()

# ============================
# 5. Preprocesadores
# ============================
preprocess_tree = ColumnTransformer(transformers=[
    ("num", "passthrough", selected_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     selected_categorical)
])

preprocess_linear = ColumnTransformer(transformers=[
    ("num", StandardScaler(), selected_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     selected_categorical)
])

X_tree   = preprocess_tree.fit_transform(X)
X_linear = preprocess_linear.fit_transform(X)

ohe_names = (preprocess_tree.named_transformers_["cat"]
             .get_feature_names_out(selected_categorical))
all_feature_names = np.array(selected_numeric + list(ohe_names))

consensus_features = [
    "ESM1b_converted_rankscore", "PrimateAI_rankscore",
    "SIFT4G_converted_rankscore", "Posicion",
    "Dominio_agrupado_dii", "Dominio_agrupado_diii_div_linker",
    "Alelo_Referencia_g", "Alelo_Alternativo_c", "Alelo_Alternativo_g"
]

selected_mask = np.isin(all_feature_names, consensus_features)
X_tree   = X_tree[:, selected_mask]
X_linear = X_linear[:, selected_mask]

# ============================
# 6. Métricas personalizadas
# ============================
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def npv_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fn = cm[0, 0], cm[1, 0]
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def brier_scorer(estimator, X, y):
    y_proba = estimator.predict_proba(X)[:, 1]
    return -brier_score_loss(y, y_proba)

scorers = {
    "AUC-ROC":     "roc_auc",
    "F1 macro":    make_scorer(f1_score, average="macro"),
    "MCC":         make_scorer(matthews_corrcoef),
    "Sensitivity": make_scorer(f1_score, pos_label=1, average="binary",
                               zero_division=0),
    "Specificity": make_scorer(specificity_score),
    "NPV":         make_scorer(npv_score),
    "Brier score": brier_scorer
}

# ============================
# 7. Definir modelos
# ============================
cv        = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_single = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,  random_state=42)

models = {
    "XGBoost": (
        xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbosity=0
        ),
        X_tree
    ),
    "LightGBM": (
        lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            objective="binary", class_weight="balanced",
            random_state=42, verbose=-1
        ),
        X_tree
    ),
    "CatBoost": (
        CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=4,
            loss_function="Logloss",
            class_weights=[1, scale_pos_weight],
            random_seed=42, verbose=0
        ),
        X_tree
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300, max_depth=4,
            class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        X_linear
    ),
    "SVM": (
        SVC(
            kernel="rbf", probability=True,
            class_weight="balanced",
            random_state=42
        ),
        X_linear
    ),
    "Reg. logística (L2)": (
        LogisticRegression(
            penalty="l2", solver="lbfgs",
            class_weight="balanced",
            max_iter=1000, random_state=42
        ),
        X_linear
    ),
    "Reg. logística (L1)": (
        LogisticRegression(
            penalty="l1", solver="liblinear",
            class_weight="balanced",
            max_iter=1000, random_state=42
        ),
        X_linear
    ),
}

# ============================
# 8. Evaluar modelos
# ============================
print("\n" + "=" * 95)
print("COMPARACIÓN DE MODELOS (RepeatedStratifiedKFold 5x10)")
print("=" * 95)

results  = {metric: {} for metric in scorers}
y_probas = {}

for name, (model, X_data) in models.items():
    print(f"Evaluando {name}...")
    for metric_name, scorer in scorers.items():
        scores = cross_val_score(model, X_data, y,
                                 cv=cv, scoring=scorer, n_jobs=-1)
        results[metric_name][name] = scores

    y_probas[name] = cross_val_predict(
        model, X_data, y, cv=cv_single, method="predict_proba"
    )[:, 1]

# ============================
# 9. Tabla de resultados
# ============================
table_rows = []
for name in models.keys():
    row = {"Modelo": name}
    for metric_name in scorers.keys():
        scores = results[metric_name][name]
        if metric_name == "Brier score":
            row[metric_name] = f"{abs(scores.mean()):.3f} ± {scores.std():.3f}"
        else:
            row[metric_name] = f"{scores.mean():.3f} ± {scores.std():.3f}"
    table_rows.append(row)

results_df = pd.DataFrame(table_rows).set_index("Modelo")
print("\n" + results_df.to_string())

csv_path = os.path.join(RES_DIR, "tabla_comparacion_modelos.csv")
results_df.to_csv(csv_path)
print(f"\nTabla guardada: {csv_path}")

# ============================
# 10. Visualización — Curvas ROC
# ============================
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

color_palette = [
    "#E74C3C",  # XGBoost — rojo
    "#3498DB",  # LightGBM — azul
    "#2ECC71",  # CatBoost — verde
    "#9B59B6",  # Random Forest — morado
    "#F39C12",  # SVM — naranja
    "#1ABC9C",  # Reg. logística L2 — turquesa
    "#34495E",  # Reg. logística L1 — gris oscuro
]

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
        linewidth=1, alpha=0.5, label="Clasificador aleatorio")

for (name, y_proba), color in zip(y_probas.items(), color_palette):
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    lw = 2.5 if name == "Reg. logística (L2)" else 1.5
    ax.plot(fpr, tpr, color=color, linewidth=lw,
            label=f"{name} (AUC = {roc_auc:.3f})")

ax.set_title("Curvas ROC — Comparación de modelos", fontweight="bold")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
roc_path = os.path.join(FIG_DIR, "figura_roc_comparacion.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print(f"\nFigura guardada: {roc_path}")
