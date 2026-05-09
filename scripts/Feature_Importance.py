"""
Selección de características para variantes missense de SCN1A mediante:
- RFECV (XGBoost)
- SelectFromModel (XGBoost)
- Permutation Importance
y combinación por consenso (≥2/3).

El script:
1. Carga el dataset procesado.
2. Limpia y normaliza variables categóricas.
3. Codifica variables con OneHotEncoder.
4. Ejecuta tres métodos de selección de características.
5. Combina resultados por votación.
6. Evalúa el modelo final con CV.
7. Genera figuras de calidad para el TFM.

Requiere:
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================
# 1. Rutas robustas
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "..", "data", "SCN1A_ML_complete_dbnsfp.xlsx")
FIG_DIR = os.path.join(BASE_DIR, "..", "figures")

os.makedirs(FIG_DIR, exist_ok=True)

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

print(f"Distribución — LoF: {(y==0).sum()}, GoF: {(y==1).sum()}")

# ============================
# 4. Features
# ============================
numeric_features = [
    "VARITY_ER_rankscore", "ESM1b_converted_rankscore", "AlphaMissense_rankscore",
    "MutPred2_rankscore", "MPC_rankscore", "PrimateAI_rankscore",
    "REVEL_rankscore", "MetaRNN_rankscore", "SIFT4G_converted_rankscore",
    "Polyphen2_HDIV_rankscore", "Posicion"
]
categorical_features = ["Dominio_agrupado", "Alelo_Referencia", "Alelo_Alternativo"]

X = df[numeric_features + categorical_features].copy()

# ============================
# 5. Preprocesador
# ============================
preprocess = ColumnTransformer(transformers=[
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_features)
])

X_proc = preprocess.fit_transform(X)

ohe_names = (preprocess.named_transformers_["cat"]
             .get_feature_names_out(categorical_features))
all_feature_names = np.array(numeric_features + list(ohe_names))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)

# ============================
# 6. MÉTODO 1 — RFECV
# ============================
print("Ejecutando RFECV...")
rfecv = RFECV(
    estimator=base_clf,
    step=1,
    cv=cv,
    scoring="roc_auc",
    min_features_to_select=5,
    n_jobs=-1
)
rfecv.fit(X_proc, y)
features_rfecv = set(all_feature_names[rfecv.support_])

# ============================
# 7. MÉTODO 2 — SelectFromModel
# ============================
print("Ejecutando SelectFromModel...")
sfm_clf = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", random_state=42, verbosity=0
)
sfm_clf.fit(X_proc, y)

sfm = SelectFromModel(sfm_clf, prefit=True, threshold="mean")
features_sfm = set(all_feature_names[sfm.get_support()])

# ============================
# 8. MÉTODO 3 — Permutation Importance
# ============================
print("Ejecutando Permutation Importance...")
perm_clf = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", random_state=42, verbosity=0
)
perm_clf.fit(X_proc, y)

perm_result = permutation_importance(
    perm_clf, X_proc, y,
    n_repeats=20,
    random_state=42,
    n_jobs=-1,
    scoring="roc_auc"
)
perm_threshold = np.mean(perm_result.importances_mean)
features_perm = set(all_feature_names[perm_result.importances_mean > perm_threshold])

# ============================
# 9. Consenso (≥2 de 3)
# ============================
all_candidates = features_rfecv | features_sfm | features_perm
feature_votes = {
    feat: sum([
        feat in features_rfecv,
        feat in features_sfm,
        feat in features_perm
    ])
    for feat in all_candidates
}

selected_features = [f for f, v in feature_votes.items() if v >= 2]
selected_mask = np.isin(all_feature_names, selected_features)
X_selected = X_proc[:, selected_mask]
selected_names = all_feature_names[selected_mask]

# ============================
# 10. Modelo final
# ============================
final_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)

scores = cross_val_score(final_clf, X_selected, y, cv=cv, scoring="roc_auc")
scores_all = cross_val_score(base_clf, X_proc, y, cv=cv, scoring="roc_auc")

final_clf.fit(X_selected, y)

fi_final = pd.DataFrame({
    "feature": selected_names,
    "importance": final_clf.feature_importances_
}).sort_values("importance", ascending=False)

# ============================
# 11. Visualizaciones
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

orange_sel      = "#E07B39"
orange_disc     = "#F2C9A8"
cmap_importance = "YlOrBr"

vote_df = (pd.DataFrame.from_dict(feature_votes, orient="index", columns=["votos"])
           .reset_index().rename(columns={"index": "feature"})
           .sort_values("votos", ascending=True))

# ============================
# FIGURA 1 — Votos
# ============================
fig1, ax1 = plt.subplots(figsize=(8, len(vote_df) * 0.35))
colors = [orange_sel if v >= 2 else orange_disc for v in vote_df["votos"]]

ax1.barh(vote_df["feature"], vote_df["votos"], color=colors, edgecolor="white")
ax1.axvline(x=2, color="#7A3B0A", linestyle="--")
ax1.set_title("Votos por característica")
ax1.set_xlabel("Número de métodos que la seleccionan")

# LEYENDA RESTAURADA
patch_sel  = mpatches.Patch(color=orange_sel,  label="Seleccionada")
patch_disc = mpatches.Patch(color=orange_disc, label="Descartada")
ax1.legend(handles=[patch_sel, patch_disc], fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figura_votos.png"), dpi=300, bbox_inches="tight")
plt.show()

# ============================
# FIGURA 2 — Importancia + RFECV
# ============================
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Selección de características", fontsize=14, fontweight="bold")

# --- Importancia XGBoost ---
fi_sorted = fi_final.sort_values("importance", ascending=True)
palette = sns.color_palette(cmap_importance, len(fi_sorted))

axes[0].barh(fi_sorted["feature"], fi_sorted["importance"], color=palette)
axes[0].set_title("Importancia XGBoost - SelectFromModel")
axes[0].set_xlabel("Importancia")

# MEDIA RESTAURADA
mean_importance = fi_sorted["importance"].mean()
axes[0].axvline(
    x=mean_importance,
    color="#7A3B0A",
    linestyle="--",
    linewidth=1.2,
    label=f"Media = {mean_importance:.3f}"
)
axes[0].legend(fontsize=9, frameon=False)

# --- RFECV ---
n_features_range = list(range(
    rfecv.min_features_to_select,
    len(rfecv.cv_results_["mean_test_score"]) + rfecv.min_features_to_select
))
mean_scores = rfecv.cv_results_["mean_test_score"]
std_scores  = rfecv.cv_results_["std_test_score"]

axes[1].plot(n_features_range, mean_scores, color=orange_sel, marker="o")
axes[1].fill_between(
    n_features_range,
    mean_scores - std_scores,
    mean_scores + std_scores,
    color=orange_sel, alpha=0.2
)
axes[1].axvline(x=rfecv.n_features_, color="#7A3B0A", linestyle="--")
axes[1].set_title("RFECV: Número de características vs AUC-ROC")
axes[1].set_xlabel("Número de características")
axes[1].set_ylabel("AUC-ROC")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figura_importancia_rfecv.png"), dpi=300, bbox_inches="tight")
plt.show()
