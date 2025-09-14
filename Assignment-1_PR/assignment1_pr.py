# assignment1_pr.py
# Name- Shashi Suman (2511CS13)
#Submitted to : Dr. Chandranath Adak Sir
# Requirements: scikit-learn, pandas, matplotlib, seaborn, numpy


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names


df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df.to_csv(os.path.join(OUTDIR, "wine_head.csv"), index=False)

# Train/Test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)
explained = pca.explained_variance_ratio_

# Save PCA info
with open(os.path.join(OUTDIR, "pca_info.txt"), "w") as f:
    f.write(f"Explained variance ratios (2 comps): {explained}\n")
    f.write(f"Total explained (2 comps): {explained.sum():.4f}\n")

# Plot PCA scatter (train) 
plt.figure(figsize=(7,5))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=y_train, palette="deep", s=60)
plt.title("Wine dataset - PCA (2 components) - TRAIN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="class")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pca_train_scatter.png"), dpi=200)
plt.close()

results = {}

# Logistic Regression on ORIGINAL standardized features
lr_orig = LogisticRegression(max_iter=1000, random_state=42)
lr_orig.fit(X_train_s, y_train)
yhat_lr_orig = lr_orig.predict(X_test_s)
acc_lr_orig = accuracy_score(y_test, yhat_lr_orig)
results['Logistic_Original'] = (acc_lr_orig, classification_report(y_test, yhat_lr_orig, target_names=class_names), confusion_matrix(y_test, yhat_lr_orig))

# Logistic Regression on PCA-reduced features
lr_pca = LogisticRegression(max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train)
yhat_lr_pca = lr_pca.predict(X_test_pca)
acc_lr_pca = accuracy_score(y_test, yhat_lr_pca)
results['Logistic_PCA2'] = (acc_lr_pca, classification_report(y_test, yhat_lr_pca, target_names=class_names), confusion_matrix(y_test, yhat_lr_pca))

# k-NN with Euclidean (p=2)
knn_euc = KNeighborsClassifier(n_neighbors=5, p=2)  
knn_euc.fit(X_train_s, y_train)
yhat_knn_euc = knn_euc.predict(X_test_s)
acc_knn_euc = accuracy_score(y_test, yhat_knn_euc)
results['kNN_Euclidean_p5'] = (acc_knn_euc, classification_report(y_test, yhat_knn_euc, target_names=class_names), confusion_matrix(y_test, yhat_knn_euc))

# k-NN with Manhattan (p=1)
knn_man = KNeighborsClassifier(n_neighbors=5, p=1)  
knn_man.fit(X_train_s, y_train)
yhat_knn_man = knn_man.predict(X_test_s)
acc_knn_man = accuracy_score(y_test, yhat_knn_man)
results['kNN_Manhattan_p5'] = (acc_knn_man, classification_report(y_test, yhat_knn_man, target_names=class_names), confusion_matrix(y_test, yhat_knn_man))

# Optional: k-NN in PCA space (Euclidean)
knn_pca = KNeighborsClassifier(n_neighbors=5, p=2)
knn_pca.fit(X_train_pca, y_train)
yhat_knn_pca = knn_pca.predict(X_test_pca)
acc_knn_pca = accuracy_score(y_test, yhat_knn_pca)
results['kNN_PCA2'] = (acc_knn_pca, classification_report(y_test, yhat_knn_pca, target_names=class_names), confusion_matrix(y_test, yhat_knn_pca))

# Save results and confusion matrices
summary_lines = []
for name, (acc, crep, cm) in results.items():
    summary_lines.append(f"--- {name} ---")
    summary_lines.append(f"Accuracy: {acc:.4f}")
    summary_lines.append("Classification Report:")
    summary_lines.append(crep)
    summary_lines.append("Confusion Matrix:")
    summary_lines.append(np.array2string(cm))
    summary_lines.append("\n")

    # plot confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fname = os.path.join(OUTDIR, f"cm_{name}.png")
    plt.savefig(fname, dpi=200)
    plt.close()

with open(os.path.join(OUTDIR, "summary.txt"), "w") as f:
    f.write("Wine dataset - Model comparison (results)\n\n")
    f.write("\n".join(summary_lines))

# Save a small textual report snippet about bias-variance
bv_text = """
Bias-Variance notes (short):
- Logistic Regression tends to have lower variance and can be biased if model is not flexible enough to separate classes.
- k-NN with k small (e.g., 1) -> low bias, high variance; with larger k -> higher bias, lower variance.
- PCA reduces dimensionality and may reduce variance (helps with overfitting) but may increase bias if informative features are discarded.
"""
with open(os.path.join(OUTDIR, "bias_variance_notes.txt"), "w") as f:
    f.write(bv_text)

print("All outputs saved to folder:", OUTDIR)
print("Files produced:", os.listdir(OUTDIR))
