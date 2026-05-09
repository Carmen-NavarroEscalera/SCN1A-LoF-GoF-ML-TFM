"""
Modelo final basado en Regresión Logística (L2) optimizada para clasificar variantes
LoF vs GoF en SCN1A.

Incluye:
- Preprocesado de variables numéricas y categóricas
- Selección de características por consenso
- Optimización de hiperparámetros con GridSearchCV
- Evaluación con validación cruzada (ROC, PR, Brier, Youden)
- Cálculo de valores SHAP
- Curva de aprendizaje
- Figuras de evaluación estilo TFM

Requiere:
- pandas, numpy
- scikit-learn
- shap
- matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, learning_curve,
                                     GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, brier_score_loss)
import shap
import matplotlib.pyplot as plt
import seaborn as sns
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
print(f"Distribución de clases — LoF: {n_lof}, GoF: {n_gof}")

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
# 5. Preprocesador
# ============================
preprocess = ColumnTransformer(transformers=[
    ("num", StandardScaler(), selected_numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     selected_categorical)
])

X_proc = preprocess.fit_transform(X)

ohe_names = (preprocess.named_transformers_["cat"]
             .get_feature_names_out(selected_categorical))
all_feature_names = np.array(selected_numeric + list(ohe_names))

consensus_features = [
    "ESM1b_converted_rankscore", "PrimateAI_rankscore",
    "SIFT4G_converted_rankscore", "Posicion",
    "Dominio_agrupado_dii", "Dominio_agrupado_diii_div_linker",
    "Alelo_Referencia_g", "Alelo_Alternativo_c", "Alelo_Alternativo_g"
]

selected_mask = np.isin(all_feature_names, consensus_features)
X_selected = X_proc[:, selected_mask]
selected_names = all_feature_names[selected_mask]

# ============================
# 6. CV interna y externa
# ============================
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# ============================
# 7. GridSearchCV
# ============================
print("\nEjecutando búsqueda exhaustiva de hiperparámetros (GridSearchCV)...")

param_grid = {
    "C":        [0.001, 0.01, 0.1, 1, 10, 100],
    "solver":   ["lbfgs", "newton-cg"],
    "max_iter": [500, 1000, 2000]
}

search_clf = LogisticRegression(
    penalty="l2",
    class_weight="balanced",
    random_state=42
)

grid_search = GridSearchCV(
    search_clf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv_inner,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_selected, y)

print(f"\nMejores hiperparámetros: {grid_search.best_params_}")
print(f"AUC-ROC óptimo (CV):     {grid_search.best_score_:.3f}")

best_params = grid_search.best_params_

# ============================
# 8. Modelo final
# ============================
final_clf = LogisticRegression(
    penalty="l2",
    class_weight="balanced",
    random_state=42,
    **best_params
)

y_pred  = cross_val_predict(final_clf, X_selected, y,
                            cv=cv_inner, method="predict")
y_proba = cross_val_predict(final_clf, X_selected, y,
                            cv=cv_inner, method="predict_proba")[:, 1]

print("\n" + "=" * 50)
print("MÉTRICAS DE CLASIFICACIÓN (umbral = 0.5)")
print("=" * 50)
print(classification_report(y, y_pred, target_names=["LoF", "GoF"], digits=3))

fpr, tpr, thresholds = roc_curve(y, y_proba)
roc_auc = auc(fpr, tpr)
print(f"AUC-ROC: {roc_auc:.3f}")

precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)
avg_precision = average_precision_score(y, y_proba)
baseline_pr = n_gof / (n_lof + n_gof)
print(f"AUC-PR:  {avg_precision:.3f} (baseline aleatorio: {baseline_pr:.2f})")

brier = brier_score_loss(y, y_proba)
print(f"Brier score: {brier:.3f}")

# ============================
# 9. Umbral óptimo (Youden)
# ============================
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

print(f"\nUmbral óptimo (índice de Youden): {optimal_threshold:.3f}")
print(f"  Sensitivity: {optimal_sensitivity:.3f}")
print(f"  Specificity: {optimal_specificity:.3f}")

y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
print("\nMétricas con umbral óptimo:")
print(classification_report(y, y_pred_optimal, target_names=["LoF", "GoF"], digits=3))

final_clf.fit(X_selected, y)

# ============================
# 10. SHAP
# ============================
print("\nCalculando valores SHAP...")
explainer   = shap.LinearExplainer(final_clf, X_selected)
shap_values = explainer.shap_values(X_selected)

# ============================
# 11. Curva de aprendizaje
# ============================
print("Calculando curva de aprendizaje...")
train_sizes, train_scores, val_scores = learning_curve(
    final_clf, X_selected, y,
    cv=cv_outer,
    scoring="roc_auc",
    train_sizes=np.linspace(0.2, 1.0, 8),
    n_jobs=-1
)

# ============================
# 12. Visualizaciones
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

orange_main = "#E07B39"
dark_orange  = "#7A3B0A"

# Figura 1 — Matriz de confusión + ROC + PR
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle("Evaluación del modelo final — SCN1A",
              fontsize=14, fontweight="bold", y=1.02)

cm = confusion_matrix(y, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlOrBr",
    xticklabels=["LoF", "GoF"],
    yticklabels=["LoF", "GoF"],
    ax=axes1[0], linewidths=0.5, linecolor="white",
    cbar=False, annot_kws={"size": 14}
)
axes1[0].set_title("Matriz de confusión")
axes1[0].set_xlabel("Clase predicha")
axes1[0].set_ylabel("Clase real")

axes1[1].plot(fpr, tpr, color=orange_main, linewidth=2,
              label=f"AUC-ROC = {roc_auc:.3f}")
axes1[1].plot([0, 1], [0, 1], color=dark_orange, linestyle="--",
              linewidth=1, label="Clasificador aleatorio")
axes1[1].scatter(fpr[optimal_idx], tpr[optimal_idx],
                 color=dark_orange, s=80, zorder=5,
                 label=f"Umbral óptimo = {optimal_threshold:.3f}")
axes1[1].set_title("Curva ROC")
axes1[1].set_xlabel("Tasa de falsos positivos")
axes1[1].set_ylabel("Tasa de verdaderos positivos")
axes1[1].legend(fontsize=9, frameon=False)

axes1[2].plot(recall_vals, precision_vals, color=orange_main,
              linewidth=2, label=f"AUC-PR = {avg_precision:.3f}")
axes1[2].axhline(y=baseline_pr, color=dark_orange, linestyle="--",
                 linewidth=1, label=f"Baseline = {baseline_pr:.2f}")
axes1[2].set_title("Curva precisión-recall")
axes1[2].set_xlabel("Recall")
axes1[2].set_ylabel("Precisión")
axes1[2].legend(fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figura_evaluacion.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.show()

# ============================
# Figura 2 — SHAP (con ejes añadidos)
# ============================
fig2, ax2 = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_selected,
    feature_names=selected_names,
    show=False,
    plot_type="dot"
)

# Añadir ejes personalizados
ax2.set_xlabel("Impacto en la predicción (valor SHAP)", fontsize=12)
ax2.set_ylabel("Características", fontsize=12)

plt.title("Valores SHAP — Contribución de cada característica",
          fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figura_shap.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.show()

# ============================
# Figura 3 — Curva de aprendizaje
# ============================
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(train_sizes, train_mean, color=orange_main,
         linewidth=2, marker="o", markersize=4, label="Entrenamiento")
ax3.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std,
                 color=orange_main, alpha=0.2)
ax3.plot(train_sizes, val_mean, color=dark_orange,
         linewidth=2, marker="o", markersize=4,
         linestyle="--", label="Validación cruzada")
ax3.fill_between(train_sizes,
                 val_mean - val_std,
                 val_mean + val_std,
                 color=dark_orange, alpha=0.2)
ax3.set_title("Curva de aprendizaje", fontweight="bold")
ax3.set_xlabel("Tamaño del conjunto de entrenamiento")
ax3.set_ylabel("AUC-ROC")
ax3.legend(fontsize=9, frameon=False)
ax3.set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figura_curva_aprendizaje.png"), dpi=300,
            bbox_inches="tight", facecolor="white")
plt.show()

print("\nScript completado.")
print("Figuras guardadas en:", FIG_DIR)
