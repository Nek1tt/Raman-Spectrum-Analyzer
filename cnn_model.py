"""
cnn_model.py — CNN architecture (2-channel ResNet+SE), data augmentation
dataset, and CNNTrainer (with optional Optuna hyper-parameter search).

Public API
----------
build_cnn_model(n_grid, n_classes, torch_nn, dropout) → nn.Module
RamanDataset                                           — PyTorch Dataset
CNNTrainer                                             — fit / predict / save / load
try_import_torch()                                     — safe import helper
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Torch import helper
# ---------------------------------------------------------------------------

def try_import_torch() -> Tuple:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, optim, DataLoader, TensorDataset
    except ImportError:
        print("  [WARN] PyTorch not installed. CNN will be skipped.")
        return None, None, None, None, None


def _try_import_torch_dataset():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset
        return torch, F, Dataset
    except ImportError:
        return None, None, None


# ---------------------------------------------------------------------------
# CNN architecture
# ---------------------------------------------------------------------------

def build_cnn_model(n_grid: int, n_classes: int, torch_nn, dropout: float = 0.4):
    """
    2-channel 1-D ResNet with Squeeze-and-Excitation blocks.
    dropout is a parameter so Optuna can tune it.
    """
    nn = torch_nn

    class SEBlock1d(nn.Module):
        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc   = nn.Sequential(
                nn.Linear(channels, max(channels // reduction, 1)),
                nn.ReLU(),
                nn.Linear(max(channels // reduction, 1), channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            s = self.pool(x).squeeze(-1)
            s = self.fc(s).unsqueeze(-1)
            return x * s

    class ResBlock1d(nn.Module):
        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            kernel: int = 7,
            stride: int = 1,
        ) -> None:
            super().__init__()
            pad = kernel // 2
            self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                          padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            self.se       = SEBlock1d(out_ch)
            self.relu     = nn.ReLU()
            self.dropout  = nn.Dropout(0.2)
            self.shortcut = (
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch),
                )
                if (in_ch != out_ch or stride != 1)
                else nn.Identity()
            )

        def forward(self, x):
            out = self.conv(x)
            out = self.se(out)
            out = self.relu(out + self.shortcut(x))
            return self.dropout(out)

    class RamanResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.layer1 = ResBlock1d(32,  64,  kernel=7, stride=2)
            self.layer2 = ResBlock1d(64,  128, kernel=5, stride=2)
            self.layer3 = ResBlock1d(128, 128, kernel=3, stride=1)
            self.gap    = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.gap(x)
            return self.classifier(x)

    return RamanResNet()


# ---------------------------------------------------------------------------
# Data augmentation dataset
# ---------------------------------------------------------------------------

class RamanDataset:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = True,
        shift_max: int = 3,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> None:
        torch_mod, self._F, Dataset = _try_import_torch_dataset()
        if torch_mod is None:
            raise ImportError("PyTorch required for RamanDataset")
        self._torch    = torch_mod
        self.X         = torch_mod.FloatTensor(X)
        self.y         = torch_mod.LongTensor(y)
        self.augment   = augment
        self.shift_max = shift_max
        self.noise_std = noise_std
        self.scale_lo, self.scale_hi = scale_range

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        if self.augment:
            shift = self._torch.randint(0, self.shift_max + 1, (1,)).item()
            if self._torch.rand(1).item() > 0.5 and shift > 0:
                x = self._F.pad(x[..., shift:], (0, shift), mode="replicate")
            elif shift > 0:
                x = self._F.pad(x[..., :-shift], (shift, 0), mode="replicate")
            x = x + self._torch.randn_like(x) * self.noise_std
            scale = self._torch.FloatTensor(1).uniform_(
                self.scale_lo, self.scale_hi
            ).item()
            x = x * scale
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# CNN Trainer
# ---------------------------------------------------------------------------

class CNNTrainer:
    """
    Wraps build_cnn_model + fit/predict/save/load.

    If optuna_n_trials > 0, runs a fast Optuna search over
    (lr, weight_decay, dropout) before the final training run.
    """

    def __init__(
        self,
        n_grid: int,
        n_classes: int,
        device,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.4,
        patience: int = 10,
        optuna_n_trials: int = 0,
        optuna_epochs: int = 10,
    ) -> None:
        self.n_grid          = n_grid
        self.n_classes       = n_classes
        self.device          = device
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.dropout         = dropout
        self.patience        = patience
        self.optuna_n_trials = optuna_n_trials
        self.optuna_epochs   = optuna_epochs
        self.model           = None
        self._torch_mods     = try_import_torch()
        self.best_params: Dict[str, Any] = {}

    def _get_torch(self):
        return self._torch_mods

    # ------------------------------------------------------------------
    # Optuna search
    # ------------------------------------------------------------------

    def _optuna_search(
        self,
        X_pix: np.ndarray,
        y_pix: np.ndarray,
        class_weights: Optional[List[float]],
    ) -> Dict[str, Any]:
        if not OPTUNA_AVAILABLE:
            print("  [Optuna] not available, using default CNN params")
            return {}

        torch, nn, optim_mod, DataLoader, _ = self._get_torch()
        if torch is None:
            return {}

        print(f"\n  🔍 Optuna CNN search: {self.optuna_n_trials} trials "
              f"× {self.optuna_epochs} epochs ...")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(
            gss.split(X_pix, y_pix,
                      groups=np.arange(len(y_pix)) % max(1, len(y_pix) // 50))
        )
        X_tr, X_va = X_pix[tr_idx], X_pix[va_idx]
        y_tr, y_va = y_pix[tr_idx], y_pix[va_idx]

        def objective(trial):
            lr_t  = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
            wd_t  = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            do_t  = trial.suggest_float("dropout",      0.1,  0.5)

            model_t = build_cnn_model(
                self.n_grid, self.n_classes, nn, dropout=do_t
            ).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim_mod.AdamW(
                model_t.parameters(), lr=lr_t, weight_decay=wd_t
            )
            train_ds = RamanDataset(X_tr, y_tr, augment=True)
            loader   = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True)
            model_t.train()
            for _ in range(self.optuna_epochs):
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    optimizer.zero_grad()
                    nn.CrossEntropyLoss()(model_t(xb), yb).backward()
                    optimizer.step()

            model_t.eval()
            with torch.no_grad():
                X_va_t = torch.FloatTensor(X_va).to(self.device)
                preds  = model_t(X_va_t).argmax(dim=1).cpu().numpy()
            return accuracy_score(y_va, preds)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.optuna_n_trials,
                       show_progress_bar=False)

        best = study.best_params
        print(f"  ✅ Optuna CNN best: "
              f"lr={best['lr']:.2e}  "
              f"wd={best['weight_decay']:.2e}  "
              f"dropout={best['dropout']:.3f}  "
              f"val_acc={study.best_value:.3f}")
        return best

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_pix: np.ndarray,
        y_pix: np.ndarray,
        class_weights: Optional[List[float]] = None,
    ) -> "CNNTrainer":
        from tqdm import tqdm

        torch, nn, optim_mod, DataLoader, _ = self._get_torch()
        if torch is None:
            return self

        if self.optuna_n_trials > 0:
            best = self._optuna_search(X_pix, y_pix, class_weights)
            if best:
                self.lr           = best.get("lr",           self.lr)
                self.weight_decay = best.get("weight_decay", self.weight_decay)
                self.dropout      = best.get("dropout",      self.dropout)
                self.best_params  = best
                print(f"  📌 Final CNN params: lr={self.lr:.2e}  "
                      f"wd={self.weight_decay:.2e}  dropout={self.dropout:.3f}")

        self.model = build_cnn_model(
            self.n_grid, self.n_classes, nn, dropout=self.dropout
        ).to(self.device)

        train_dataset = RamanDataset(X_pix, y_pix, augment=True)
        loader = DataLoader(train_dataset, batch_size=self.batch_size,
                            shuffle=True, pin_memory=False)

        if class_weights is not None:
            cw        = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim_mod.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim_mod.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        best_loss   = float("inf")
        patience_ct = 0
        best_state  = None

        pbar = tqdm(range(self.epochs), desc="    CNN training",
                    ncols=80, leave=False)
        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(y_pix)
            scheduler.step()
            pbar.set_postfix({"loss": f"{epoch_loss:.4f}",
                              "pat":  f"{patience_ct}/{self.patience}"})
            if epoch_loss < best_loss - 1e-4:
                best_loss   = epoch_loss
                patience_ct = 0
                best_state  = {k: v.cpu().clone()
                               for k, v in self.model.state_dict().items()}
            else:
                patience_ct += 1
                if patience_ct >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_single_spectrum(
        self,
        spec_2ch: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        torch, nn, optim_mod, DataLoader, _ = self._get_torch()
        if torch is None or self.model is None:
            dummy = np.ones(self.n_classes) / self.n_classes
            return int(np.argmax(dummy)), dummy

        arr = np.asarray(spec_2ch, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3 or arr.shape[0] != 1:
            raise ValueError(
                f"predict_single_spectrum: expected (2, n_grid) or "
                f"(1, 2, n_grid), got {arr.shape}"
            )

        self.model.eval()
        with torch.no_grad():
            X_t    = torch.FloatTensor(arr).to(self.device)
            logits = self.model(X_t)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return int(np.argmax(proba)), proba

    def save(self, path: str) -> None:
        torch, *_ = self._get_torch()
        if torch is not None and self.model is not None:
            torch.save(self.model.state_dict(), path)

    def load(self, path: str, nn_module) -> None:
        torch, nn, *_ = self._get_torch()
        if torch is not None:
            self.model = build_cnn_model(
                self.n_grid, self.n_classes, nn_module,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
