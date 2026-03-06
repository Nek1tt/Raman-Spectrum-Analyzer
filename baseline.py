"""
Baseline pipeline for Raman spectra 3-class classification.
Classes: control, endo (endogenous HSP70), exo (exogenous HSP70)

No preprocessing — raw spectra features only.
Models: XGBoost (primary baseline) + SVM + 1D CNN (bonus)

Usage:
    pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
    python raman_baseline.py --data_root /path/to/dataset
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

CLASS_DIRS = {
    "control": ["mk1", "mk2a", "mk2b", "mk3"],
    "endo":    ["mend1", "mend2a", "mend2b", "mend3"],
    "exo":     ["mexo1", "mexo2a", "mexo2b", "mexo3"],
}


def load_spectrum(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single .txt spectrum file. Returns (wavenumbers, intensities)."""
    df = pd.read_csv(filepath, sep=r"\s+", comment="#",
                     names=["X", "Y", "Wave", "Intensity"])
    df = df.dropna()
    df = df.sort_values("Wave")
    return df["Wave"].values, df["Intensity"].values


def interpolate_to_grid(wavenumbers: np.ndarray,
                         intensities: np.ndarray,
                         grid: np.ndarray) -> np.ndarray:
    """Interpolate spectrum onto a common wavenumber grid."""
    return np.interp(grid, wavenumbers, intensities)


def load_dataset(data_root: str, n_grid: int = 1000):
    """
    Walk the dataset directory structure and load all spectra.
    Returns X (n_samples, n_grid), y (n_samples,), file_paths list.
    """
    data_root = Path(data_root)

    # First pass: collect all spectra and determine wavenumber range
    raw_spectra = []   # (label, wave, intensity, path)

    for label, subdirs in CLASS_DIRS.items():
        for subdir in subdirs:
            folder = data_root / label / label / subdir  # e.g. control/control/mk1
            if not folder.exists():
                # try one level up
                folder = data_root / label / subdir
            if not folder.exists():
                print(f"  [WARN] folder not found: {folder}")
                continue
            txt_files = list(folder.glob("*.txt"))
            print(f"  {label}/{subdir}: {len(txt_files)} files")
            for fpath in txt_files:
                try:
                    wave, intensity = load_spectrum(str(fpath))
                    if len(wave) > 10:
                        raw_spectra.append((label, wave, intensity, str(fpath)))
                except Exception as e:
                    print(f"    [SKIP] {fpath.name}: {e}")

    if not raw_spectra:
        raise ValueError("No spectra loaded! Check --data_root path.")

    # Build common grid from global min/max wavenumber
    all_waves = np.concatenate([s[1] for s in raw_spectra])
    w_min, w_max = all_waves.min(), all_waves.max()
    grid = np.linspace(w_min, w_max, n_grid)
    print(f"\nWavenumber grid: {w_min:.1f} – {w_max:.1f} cm⁻¹  ({n_grid} points)")

    # Second pass: interpolate onto grid
    X, y, paths = [], [], []
    for label, wave, intensity, fpath in raw_spectra:
        interp = interpolate_to_grid(wave, intensity, grid)
        X.append(interp)
        y.append(label)
        paths.append(fpath)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Dataset: {X.shape[0]} spectra, {X.shape[1]} features")
    print(f"Class distribution: { {c: (y==c).sum() for c in np.unique(y)} }")
    return X, y, grid, paths


# ─────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────

def build_xgboost(n_classes=3):
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def build_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    decision_function_shape="ovr", random_state=42)),
    ])


# ─────────────────────────────────────────────
# 3. OPTIONAL: 1D CNN (PyTorch)
# ─────────────────────────────────────────────

def build_cnn(n_features: int, n_classes: int = 3):
    try:
        import torch
        import torch.nn as nn

        class SpectralCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=15, padding=7), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=9, padding=4), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(32),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 32, 256), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(256, n_classes),
                )

            def forward(self, x):
                return self.classifier(self.encoder(x))

        return SpectralCNN()
    except ImportError:
        print("[INFO] PyTorch not available, skipping CNN.")
        return None


def train_cnn(model, X: np.ndarray, y_enc: np.ndarray,
              epochs: int = 50, batch_size: int = 16):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.model_selection import train_test_split

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_t, y_t, test_size=0.2, stratify=y_enc, random_state=42)

        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val_acc, best_state = 0, None
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val.to(device)).argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_val.numpy(), val_preds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  val_acc={val_acc:.3f}")

        model.load_state_dict(best_state)
        print(f"  Best CNN val_acc: {best_val_acc:.3f}")
        return model

    except ImportError:
        return None


def predict_cnn(model, X: np.ndarray):
    try:
        import torch
        device = next(model.parameters()).device
        X_t = torch.tensor(X[:, np.newaxis, :], dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            return model(X_t).argmax(1).cpu().numpy()
    except Exception:
        return None


# ─────────────────────────────────────────────
# 4. EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate_cv(model, X, y, cv=5):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_enc, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    return scores, le


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(importances, grid, top_n=30, save_path=None):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(grid[idx], importances[idx], width=5, color="steelblue")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Feature importance")
    ax.set_title(f"Top-{top_n} informative spectral regions (XGBoost)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_mean_spectra(X, y, grid, save_path=None):
    classes = np.unique(y)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for cls, color in zip(classes, colors):
        mask = y == cls
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        ax.plot(grid, mean, label=cls, color=color)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Mean ± std Raman spectra per class (no preprocessing)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root folder containing control/, endo/, exo/")
    parser.add_argument("--n_grid", type=int, default=1000,
                        help="Number of interpolation points (default 1000)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds (default 5)")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to disk instead of showing")
    parser.add_argument("--run_cnn", action="store_true",
                        help="Also train 1D CNN (requires PyTorch)")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # ── Load data ────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    X, y, grid, paths = load_dataset(args.data_root, n_grid=args.n_grid)

    # ── Visualise mean spectra ────────────────
    save_path = str(out_dir / "mean_spectra.png") if args.save_plots else None
    plot_mean_spectra(X, y, grid, save_path=save_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"\nClasses: {classes}")

    # ── XGBoost ──────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 1: XGBoost (baseline)")
    print("=" * 60)
    xgb_model = build_xgboost()
    scores, _ = evaluate_cv(xgb_model, X, y, cv=args.cv)
    print(f"  {args.cv}-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Full fit for CM + feature importance
    xgb_model.fit(X, y_enc)
    y_pred_xgb = xgb_model.predict(X)
    print("\nClassification report (train set):")
    print(classification_report(y_enc, y_pred_xgb, target_names=classes))

    save_path = str(out_dir / "cm_xgboost.png") if args.save_plots else None
    plot_confusion_matrix(y_enc, y_pred_xgb, classes,
                          "XGBoost – Train Confusion Matrix", save_path)

    save_path = str(out_dir / "feature_importance.png") if args.save_plots else None
    plot_feature_importance(xgb_model.feature_importances_, grid,
                            save_path=save_path)

    # ── SVM ──────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: SVM RBF (baseline)")
    print("=" * 60)
    svm_model = build_svm()
    scores_svm, _ = evaluate_cv(svm_model, X, y, cv=args.cv)
    print(f"  {args.cv}-fold CV accuracy: {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")

    # ── CNN (optional) ───────────────────────
    if args.run_cnn:
        print("\n" + "=" * 60)
        print("MODEL 3: 1D CNN (PyTorch)")
        print("=" * 60)
        cnn = build_cnn(n_features=args.n_grid)
        if cnn is not None:
            cnn = train_cnn(cnn, X, y_enc)
            if cnn is not None:
                y_pred_cnn = predict_cnn(cnn, X)
                print("\nCNN Classification report (train set):")
                print(classification_report(y_enc, y_pred_cnn,
                                            target_names=classes))
                save_path = (str(out_dir / "cm_cnn.png")
                             if args.save_plots else None)
                plot_confusion_matrix(y_enc, y_pred_cnn, classes,
                                      "CNN – Train Confusion Matrix", save_path)

    # ── Summary ──────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  XGBoost CV accuracy : {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  SVM     CV accuracy : {scores_svm.mean():.3f} ± {scores_svm.std():.3f}")
    print("\nDone. Plots saved to ./outputs/" if args.save_plots
          else "\nDone.")


if __name__ == "__main__":
    main()