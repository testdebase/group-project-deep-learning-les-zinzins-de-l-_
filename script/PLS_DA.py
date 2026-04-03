"""
PLS-DA (Partial Least Squares - Discriminant Analysis)
=======================================================
Classification d'espèces végétales à partir de spectres foliaires.

Méthode purement statistique/chimiométrique :
  - Pas de réseau de neurones
  - Pas d'apprentissage itératif
  - Décomposition matricielle linéaire (algèbre linéaire)

Pipeline :b 
  1. Chargement des données
  2. Prétraitement spectral (Savitzky-Golay + StandardScaler)
  3. Choix du nombre de composantes par cross-validation (scree plot)
  4. PLS-DA : réduction de dimension supervisée
  5. Classification avec LDA sur les scores PLS
  6. Évaluation : accuracy, matrice de confusion, rapport de classification
  7. Diagnostics : scores plot, VIP (Variable Importance in Projection)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.signal import savgol_filter


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

data_src = Path(__file__).parent.parent / 'data' / 'combined_data.csv'
data = pd.read_csv(data_src)

target_col = data.columns[-1]
X = data.drop(columns=[target_col]).values    # (160, 224) — bandes spectrales
y_labels = data[target_col].values            # (160,)     — noms des espèces

le = LabelEncoder()
y = le.fit_transform(y_labels)
n_classes = len(le.classes_)
wavelengths = np.linspace(397, 1001, X.shape[1])

print("=" * 60)
print("1. DONNÉES")
print("=" * 60)
print(f"Échantillons : {X.shape[0]}, Bandes : {X.shape[1]}, Classes : {n_classes}")
for espece, count in pd.Series(y_labels).value_counts().items():
    print(f"  {espece:<20} : {count}")


# ============================================================
# 2. PRÉTRAITEMENT SPECTRAL
# ============================================================
# Savitzky-Golay : lisse les spectres et calcule la dérivée première.
# La dérivée accentue les différences entre espèces et supprime
# les effets de baseline (décalage vertical des spectres).
# StandardScaler : centre et réduit chaque bande (moyenne=0, std=1).

print("\n" + "=" * 60)
print("2. PRÉTRAITEMENT")
print("=" * 60)

X_sg = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sg)
print("Savitzky-Golay (dérivée 1ère) + StandardScaler appliqués")


# ============================================================
# 3. CHOIX DU NOMBRE DE COMPOSANTES PLS
# ============================================================
# On teste 1 à 20 composantes. Pour chaque nombre, on fait une
# cross-validation stratifiée 5-fold :
#   - on extrait les scores PLS sur le train
#   - on entraîne une LDA sur ces scores
#   - on mesure l'accuracy sur le fold de validation
# Le nombre optimal est celui qui donne la meilleure accuracy.

print("\n" + "=" * 60)
print("3. CHOIX DU NOMBRE DE COMPOSANTES (cross-validation 5-fold)")
print("=" * 60)

max_components = 20
cv_scores = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in range(1, max_components + 1):
    fold_scores = []
    for train_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # La PLSRegression attend une matrice Y continue :
        # on utilise le one-hot encoding (N×7) comme cible
        Y_tr = np.eye(n_classes)[y_tr]

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_tr, Y_tr)

        lda = LinearDiscriminantAnalysis()
        lda.fit(pls.transform(X_tr), y_tr)
        y_pred = lda.predict(pls.transform(X_val))
        fold_scores.append(accuracy_score(y_val, y_pred))

    cv_scores.append(np.mean(fold_scores))
    print(f"  {n_comp:2d} composantes : accuracy = {np.mean(fold_scores):.3f}")

best_n = np.argmax(cv_scores) + 1
print(f"\n→ Nombre optimal : {best_n} composantes (accuracy={cv_scores[best_n-1]:.3f})")


# ============================================================
# 4. MODÈLE FINAL PLS-DA
# ============================================================
# On entraîne la PLS sur toutes les données avec le nombre
# optimal de composantes, puis la LDA sur les scores obtenus.

print("\n" + "=" * 60)
print(f"4. MODÈLE FINAL PLS-DA ({best_n} composantes)")
print("=" * 60)

Y_onehot = np.eye(n_classes)[y]

pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_scaled, Y_onehot)
X_scores = pls_final.transform(X_scaled)

lda_final = LinearDiscriminantAnalysis()
lda_final.fit(X_scores, y)
y_pred = lda_final.predict(X_scores)

print(f"Scores PLS extraits : {X_scores.shape}  ({X_scores.shape[0]} échantillons, {best_n} composantes)")


# ============================================================
# 5. ÉVALUATION
# ============================================================

print("\n" + "=" * 60)
print("5. ÉVALUATION")
print("=" * 60)

# Accuracy par cross-validation (estimation non-biaisée)
cv_final_scores = []
for train_idx, val_idx in kf.split(X_scaled, y):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    Y_tr = np.eye(n_classes)[y_tr]

    pls_cv = PLSRegression(n_components=best_n)
    pls_cv.fit(X_tr, Y_tr)

    lda_cv = LinearDiscriminantAnalysis()
    lda_cv.fit(pls_cv.transform(X_tr), y_tr)
    y_pred_cv = lda_cv.predict(pls_cv.transform(X_val))
    cv_final_scores.append(accuracy_score(y_val, y_pred_cv))

print(f"Accuracy CV 5-fold : {np.mean(cv_final_scores):.3f} ± {np.std(cv_final_scores):.3f}")
print(f"Accuracy sur données complètes : {accuracy_score(y, y_pred):.3f}")
print()
print("Rapport de classification (données complètes) :")
print(classification_report(y, y_pred, target_names=le.classes_))


# ============================================================
# FIGURES → PDF
# ============================================================

pdf_path = Path(__file__).parent.parent / 'results_PLSDA.pdf'
pdf = PdfPages(pdf_path)

# --- Figure 1 : Spectres bruts ---
fig, ax = plt.subplots(figsize=(12, 5))
for cls_idx, cls_name in enumerate(le.classes_):
    idx = np.where(y == cls_idx)[0][0]
    ax.plot(wavelengths, X[idx], label=cls_name, alpha=0.8)
ax.set_xlabel("Longueur d'onde (nm)")
ax.set_ylabel("Réflectance")
ax.set_title("Spectres bruts — un échantillon par espèce")
ax.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(); plt.close()

# --- Figure 2 : Spectres après Savitzky-Golay ---
fig, ax = plt.subplots(figsize=(12, 5))
for cls_idx, cls_name in enumerate(le.classes_):
    idx = np.where(y == cls_idx)[0][0]
    ax.plot(wavelengths, X_sg[idx], label=cls_name, alpha=0.8)
ax.set_xlabel("Longueur d'onde (nm)")
ax.set_ylabel("Dérivée première")
ax.set_title("Spectres après Savitzky-Golay (dérivée 1ère)")
ax.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(); plt.close()

# --- Figure 3 : Scree plot ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, max_components + 1), cv_scores, marker='o')
ax.axvline(best_n, color='red', linestyle='--', label=f'Optimal : {best_n}')
ax.set_xlabel("Nombre de composantes PLS")
ax.set_ylabel("Accuracy (CV 5-fold)")
ax.set_title("Scree plot — Choix du nombre de composantes PLS")
ax.legend()
plt.tight_layout()
pdf.savefig(); plt.close()

# --- Figure 4 : Scores PLS (composantes 1 et 2) ---
fig, ax = plt.subplots(figsize=(8, 6))
for cls_idx, cls_name in enumerate(le.classes_):
    mask = y == cls_idx
    ax.scatter(X_scores[mask, 0], X_scores[mask, 1], label=cls_name, alpha=0.7)
ax.set_xlabel("Composante PLS 1")
ax.set_ylabel("Composante PLS 2")
ax.set_title("Scores PLS — 2 premières composantes")
ax.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(); plt.close()

# --- Figure 5 : VIP ---
def compute_vip(pls_model):
    T = pls_model.x_scores_
    W = pls_model.x_weights_
    Q = pls_model.y_loadings_
    p, h = W.shape
    vip = np.zeros(p)
    s = np.diag(T.T @ T @ Q.T @ Q)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i, j] / np.linalg.norm(W[:, j]))**2 * s[j] for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)
    return vip

vip = compute_vip(pls_final)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(wavelengths, vip, color='darkgreen')
ax.axhline(1.0, color='red', linestyle='--', label='Seuil VIP = 1')
ax.set_xlabel("Longueur d'onde (nm)")
ax.set_ylabel("VIP")
ax.set_title("Variable Importance in Projection (VIP)")
ax.legend()
plt.tight_layout()
pdf.savefig(); plt.close()

# --- Figure 6 : Matrice de confusion ---
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Prédictions")
ax.set_ylabel("Véritables")
ax.set_title("Matrice de confusion — PLS-DA")
plt.tight_layout()
pdf.savefig(); plt.close()

pdf.close()
print(f"\nPDF sauvegardé : {pdf_path}")
