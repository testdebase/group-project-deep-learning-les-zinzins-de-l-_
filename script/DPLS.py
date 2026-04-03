"""
Deep Learning Partial Least Squares (DPLS) pour la classification d'espèces végétales
=====================================================================================
Pipeline complet :
  1. Chargement et exploration des données
  2. Prétraitement spectral
  3. Détermination du nombre optimal de composantes PLS
  4. PLS-DA (réduction de dimension supervisée)
  5. Deep Learning sur les scores PLS (PyTorch)
  6. Comparaison : PLS-DA seul vs DPLS
  7. Diagnostics : VIP, scree plot, matrices de confusion

Données : combined_data.csv
  - 160 échantillons, 224 bandes spectrales (~397 nm à ~1001 nm)
  - 7 espèces : canola, kochia, ragweed, redroot_pigweed, soybean, sugarbeet, waterhemp
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = Path(__file__).parent.parent / 'results_DPLS.pdf'
pdf = PdfPages(pdf_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.signal import savgol_filter


# ============================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================
# On charge le fichier CSV et on sépare les features (X) de la cible (y).
# Les 224 premières colonnes sont les bandes spectrales,
# la dernière colonne est la classe (espèce végétale).

data_src = Path(__file__).parent.parent / 'data' / 'combined_data.csv'
data = pd.read_csv(data_src)

target_col = data.columns[-1]
X = data.drop(columns=[target_col]).values        # (160, 224) — bandes spectrales
y_labels = data[target_col].values                 # (160,)     — noms des espèces

print("=" * 60)
print("1. EXPLORATION DES DONNÉES")
print("=" * 60)
print(f"Dimensions X : {X.shape}  ({X.shape[0]} échantillons, {X.shape[1]} bandes)")
print(f"Classes ({len(np.unique(y_labels))}) :")
for espece, count in pd.Series(y_labels).value_counts().items():
    print(f"  {espece:<20} : {count} échantillons")

# Encodage de la cible en entiers (0..6)
le = LabelEncoder()
y = le.fit_transform(y_labels)
n_classes = len(le.classes_)

# Affichage des spectres bruts (un échantillon par classe)
plt.figure(figsize=(12, 5))
wavelengths = np.linspace(397, 1001, X.shape[1])
for cls_idx, cls_name in enumerate(le.classes_):
    idx = np.where(y == cls_idx)[0][0]
    plt.plot(wavelengths, X[idx], label=cls_name, alpha=0.8)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Réflectance")
plt.title("Spectres bruts — un échantillon par espèce")
plt.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(); plt.close()


# ============================================================
# 2. PRÉTRAITEMENT SPECTRAL
# ============================================================
# Deux étapes :
#   a) Filtre de Savitzky-Golay : lisse les spectres et réduit le bruit
#      instrumental. On prend la dérivée première (deriv=1) pour accentuer
#      les différences entre bandes.
#   b) StandardScaler : centre et réduit chaque bande (moyenne=0, std=1).
#      Indispensable pour que la PLS ne soit pas dominée par les bandes
#      à forte amplitude.

print("\n" + "=" * 60)
print("2. PRÉTRAITEMENT SPECTRAL")
print("=" * 60)

# a) Filtre Savitzky-Golay (fenêtre=11, polynôme ordre 2, dérivée 1ère)
X_sg = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)
print("Filtre Savitzky-Golay appliqué (dérivée 1ère, fenêtre=11)")

# b) Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sg)
print("Standardisation appliquée (moyenne=0, std=1 par bande)")


# ============================================================
# 3. DÉTERMINATION DU NOMBRE OPTIMAL DE COMPOSANTES PLS
# ============================================================
# On teste de 1 à 20 composantes et on mesure l'accuracy par
# cross-validation stratifiée (5-fold) avec un LDA sur les scores PLS.
# Le nombre optimal est celui qui maximise l'accuracy de validation.

print("\n" + "=" * 60)
print("3. NOMBRE OPTIMAL DE COMPOSANTES PLS (cross-validation)")
print("=" * 60)

max_components = 20
cv_scores = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in range(1, max_components + 1):
    fold_scores = []
    for train_idx, val_idx in kf.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Encodage one-hot de y pour la PLSRegression (nécessite une matrice Y)
        Y_tr = np.eye(n_classes)[y_tr]

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_tr, Y_tr)

        # Scores PLS du jeu de validation
        X_val_scores = pls.transform(X_val)

        # Classificateur LDA sur les scores
        lda = LinearDiscriminantAnalysis()
        lda.fit(pls.transform(X_tr), y_tr)
        y_pred = lda.predict(X_val_scores)
        fold_scores.append(accuracy_score(y_val, y_pred))

    cv_scores.append(np.mean(fold_scores))
    print(f"  {n_comp:2d} composantes : accuracy CV = {np.mean(fold_scores):.3f}")

best_n = np.argmax(cv_scores) + 1
print(f"\n→ Nombre optimal de composantes : {best_n} (accuracy={cv_scores[best_n-1]:.3f})")

# Scree plot
plt.figure(figsize=(8, 4))
plt.plot(range(1, max_components + 1), cv_scores, marker='o')
plt.axvline(best_n, color='red', linestyle='--', label=f'Optimal : {best_n}')
plt.xlabel("Nombre de composantes PLS")
plt.ylabel("Accuracy (CV 5-fold)")
plt.title("Scree plot — Choix du nombre de composantes PLS")
plt.legend()
plt.tight_layout()
pdf.savefig(); plt.close()


# ============================================================
# 4. PLS-DA : RÉDUCTION DE DIMENSION SUPERVISÉE
# ============================================================
# On entraîne la PLS sur l'ensemble des données avec le nombre
# optimal de composantes. Les scores PLS (X_scores) remplacent
# les 224 bandes spectrales — ils concentrent l'information
# discriminante entre espèces dans un espace de dimension réduite.
#
# La PLSRegression attend une matrice Y continue, donc on utilise
# le one-hot encoding (matrice N×7) comme cible.

print("\n" + "=" * 60)
print(f"4. PLS-DA AVEC {best_n} COMPOSANTES")
print("=" * 60)

Y_onehot = np.eye(n_classes)[y]   # (160, 7) — one-hot encoding

pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_scaled, Y_onehot)
X_scores = pls_final.transform(X_scaled)   # (160, best_n) — scores PLS

print(f"Scores PLS extraits : {X_scores.shape}")

# Visualisation des 2 premières composantes
plt.figure(figsize=(8, 6))
for cls_idx, cls_name in enumerate(le.classes_):
    mask = y == cls_idx
    plt.scatter(X_scores[mask, 0], X_scores[mask, 1], label=cls_name, alpha=0.7)
plt.xlabel("Composante PLS 1")
plt.ylabel("Composante PLS 2")
plt.title("Scores PLS — 2 premières composantes")
plt.legend(fontsize=8)
plt.tight_layout()
pdf.savefig(); plt.close()

# Calcul du VIP (Variable Importance in Projection)
# Le VIP mesure la contribution de chaque bande spectrale à la séparation
# des classes. Un VIP > 1 indique une bande importante.
def compute_vip(pls_model):
    T = pls_model.x_scores_          # scores
    W = pls_model.x_weights_         # poids
    Q = pls_model.y_loadings_        # loadings Y
    p, h = W.shape
    vip = np.zeros(p)
    s = np.diag(T.T @ T @ Q.T @ Q)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(W[i, j] / np.linalg.norm(W[:, j]))**2 * s[j] for j in range(h)])
        vip[i] = np.sqrt(p * np.sum(weight) / total_s)
    return vip

vip = compute_vip(pls_final)

plt.figure(figsize=(12, 4))
plt.plot(wavelengths, vip, color='darkgreen')
plt.axhline(1.0, color='red', linestyle='--', label='Seuil VIP = 1')
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("VIP")
plt.title("Variable Importance in Projection (VIP)")
plt.legend()
plt.tight_layout()
pdf.savefig(); plt.close()
print(f"Bandes spectrales importantes (VIP > 1) : {np.sum(vip > 1)} sur {len(vip)}")


# ============================================================
# 5. BASELINE : PLS-DA SEUL (sans deep learning)
# ============================================================
# On évalue d'abord les performances de la PLS-DA classique
# (PLS + LDA) par cross-validation 5-fold stratifiée.
# Ce sera notre baseline à battre avec le deep learning.

print("\n" + "=" * 60)
print("5. BASELINE : PLS-DA SEUL")
print("=" * 60)

plsda_scores = []
for train_idx, val_idx in kf.split(X_scaled, y):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    Y_tr = np.eye(n_classes)[y_tr]

    pls_cv = PLSRegression(n_components=best_n)
    pls_cv.fit(X_tr, Y_tr)

    lda = LinearDiscriminantAnalysis()
    lda.fit(pls_cv.transform(X_tr), y_tr)
    y_pred = lda.predict(pls_cv.transform(X_val))
    plsda_scores.append(accuracy_score(y_val, y_pred))

print(f"Accuracy PLS-DA (CV 5-fold) : {np.mean(plsda_scores):.3f} ± {np.std(plsda_scores):.3f}")


# ============================================================
# 6. DEEP LEARNING SUR LES SCORES PLS (DPLS)
# ============================================================
# C'est le cœur du DPLS : au lieu d'une LDA linéaire sur les scores,
# on utilise un réseau de neurones pour modéliser la relation
# (possiblement non-linéaire) entre scores PLS et classes.
#
# Architecture : scores PLS → Dense(64) → ReLU → Dropout(0.3)
#                           → Dense(32) → ReLU → Dense(7) → Softmax
#
# Avantage : entrée de dimension réduite (best_n au lieu de 224),
# ce qui limite l'overfitting sur nos 160 échantillons.

print("\n" + "=" * 60)
print("6. DEEP LEARNING SUR LES SCORES PLS (DPLS)")
print("=" * 60)

class DPLS_Net(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(DPLS_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),          # Dropout : réduit l'overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)  # Softmax implicite via CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cross-validation 5-fold sur le DPLS
dpls_scores = []
num_epochs = 150
learning_rate = 0.001
batch_size = 16

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    X_tr_raw, X_val_raw = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    Y_tr = np.eye(n_classes)[y_tr]

    # Extraction des scores PLS sur ce fold
    pls_fold = PLSRegression(n_components=best_n)
    pls_fold.fit(X_tr_raw, Y_tr)
    X_tr_scores = pls_fold.transform(X_tr_raw)
    X_val_scores = pls_fold.transform(X_val_raw)

    # Conversion en tenseurs PyTorch
    X_tr_t = torch.tensor(X_tr_scores, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_scores, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    # Initialisation du modèle
    model = DPLS_Net(best_n, n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Entraînement
    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    # Évaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_val_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_val, preds)
    dpls_scores.append(acc)
    print(f"  Fold {fold+1}/5 : accuracy = {acc:.3f}")

print(f"\nAccuracy DPLS (CV 5-fold) : {np.mean(dpls_scores):.3f} ± {np.std(dpls_scores):.3f}")


# ============================================================
# 7. COMPARAISON ET RÉSULTATS FINAUX
# ============================================================
# On entraîne le modèle final sur toutes les données pour
# visualiser la matrice de confusion, et on compare les méthodes.

print("\n" + "=" * 60)
print("7. COMPARAISON DES MÉTHODES")
print("=" * 60)

print(f"\n  PLS-DA seul : {np.mean(plsda_scores):.3f} ± {np.std(plsda_scores):.3f}")
print(f"  DPLS        : {np.mean(dpls_scores):.3f} ± {np.std(dpls_scores):.3f}")

# Barplot de comparaison
methods = ['PLS-DA', 'DPLS']
means = [np.mean(plsda_scores), np.mean(dpls_scores)]
stds = [np.std(plsda_scores), np.std(dpls_scores)]

plt.figure(figsize=(6, 4))
plt.bar(methods, means, yerr=stds, capsize=8, color=['steelblue', 'darkorange'])
plt.ylim(0, 1.1)
plt.ylabel("Accuracy (CV 5-fold)")
plt.title("Comparaison PLS-DA vs DPLS")
plt.tight_layout()
pdf.savefig(); plt.close()

# Modèle final DPLS entraîné sur toutes les données
# (pour visualiser la matrice de confusion sur le jeu complet)
Y_all = np.eye(n_classes)[y]
pls_final2 = PLSRegression(n_components=best_n)
pls_final2.fit(X_scaled, Y_all)
X_all_scores = pls_final2.transform(X_scaled)

X_all_t = torch.tensor(X_all_scores, dtype=torch.float32).to(device)
y_all_t = torch.tensor(y, dtype=torch.long).to(device)
loader_all = DataLoader(TensorDataset(X_all_t, y_all_t), batch_size=batch_size, shuffle=True)

model_final = DPLS_Net(best_n, n_classes).to(device)
optimizer_final = optim.Adam(model_final.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

model_final.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in loader_all:
        optimizer_final.zero_grad()
        loss = criterion(model_final(batch_X), batch_y)
        loss.backward()
        optimizer_final.step()

model_final.eval()
with torch.no_grad():
    y_pred_final = model_final(X_all_t).argmax(dim=1).cpu().numpy()

print("\nClassification Report (modèle final, données complètes) :")
print(classification_report(y, y_pred_final, target_names=le.classes_))

cm = confusion_matrix(y, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.title("Matrice de confusion — DPLS (modèle final)")
plt.tight_layout()
pdf.savefig(); plt.close()

pdf.close()
print(f"\nPDF sauvegardé : {pdf_path}")
