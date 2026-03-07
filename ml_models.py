"""
ml_models.py — Classical ML model definitions and Optuna-tuned Ridge.

Public API
----------
OptunaRidgeClf         — sklearn-compatible Ridge with Optuna alpha search
get_ml_models(gpu, ...) → Dict[str, estimator]
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class OptunaRidgeClf:
    """
    RidgeClassifier whose regularisation alpha is tuned by Optuna via LOGO CV.

    sklearn-compatible: fit / predict / predict_proba / get_params / set_params.
    """

    def __init__(
        self,
        n_trials: int = 20,
        cv_groups: Optional[np.ndarray] = None,
        class_weight: str = "balanced",
    ) -> None:
        self.n_trials     = n_trials
        self.cv_groups    = cv_groups
        self.class_weight = class_weight
        self.best_alpha_  = 1.0
        self._pipeline    = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptunaRidgeClf":
        if not OPTUNA_AVAILABLE or self.n_trials == 0:
            self._pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RidgeClassifier(
                    alpha=self.best_alpha_,
                    class_weight=self.class_weight
                )),
            ])
            self._pipeline.fit(X, y)
            return self

        groups = self.cv_groups
        if groups is None or len(groups) != len(y):
            groups = np.arange(len(y)) % 5

        n_splits = len(np.unique(groups))
        if n_splits > 8:
            gss      = GroupShuffleSplit(
                n_splits=min(5, n_splits), test_size=0.2, random_state=42
            )
            cv_iter  = list(gss.split(X, y, groups))
        else:
            cv_iter  = list(LeaveOneGroupOut().split(X, y, groups))

        print(f"\n  🔍 Optuna Ridge search: {self.n_trials} trials, "
              f"{len(cv_iter)} CV folds ...")

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.1, 1000.0, log=True)
            clf   = RidgeClassifier(alpha=alpha,
                                    class_weight=self.class_weight)
            accs: List[float] = []
            for tr, te in cv_iter:
                clf.fit(X_sc[tr], y[tr])
                accs.append(accuracy_score(y[te], clf.predict(X_sc[te])))
            return float(np.mean(accs))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=self.n_trials,
                       show_progress_bar=False)

        self.best_alpha_ = study.best_params["alpha"]
        print(f"  ✅ Optuna Ridge best alpha={self.best_alpha_:.4f}  "
              f"cv_acc={study.best_value:.3f}")

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RidgeClassifier(
                alpha=self.best_alpha_,
                class_weight=self.class_weight,
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._pipeline.decision_function(X)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def get_params(self, deep: bool = True) -> Dict:
        return {"n_trials": self.n_trials, "class_weight": self.class_weight}

    def set_params(self, **params) -> "OptunaRidgeClf":
        for k, v in params.items():
            setattr(self, k, v)
        return self


def get_ml_models(
    gpu: Dict[str, Any],
    optuna_ridge_trials: int = 20,
    ridge_groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Build the model catalogue.

    Parameters
    ----------
    gpu                 : GPU info dict from detect_gpu()
    optuna_ridge_trials : Optuna trials for Ridge (0 = default alpha)
    ridge_groups        : animal-id groups for LOGO inside Optuna
    """
    xgb_gpu  = (
        {"device": gpu["xgb_device"], "tree_method": gpu["xgb_tree"]}
        if gpu["available"] else {"tree_method": "hist"}
    )
    lgbm_gpu = ({"device": gpu["lgbm_device"]} if gpu["available"] else {})

    ridge_clf = OptunaRidgeClf(
        n_trials=optuna_ridge_trials,
        cv_groups=ridge_groups,
        class_weight="balanced",
    )

    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=0.1, max_iter=3000, random_state=42,
                class_weight="balanced")),
        ]),
        "RidgeClf": ridge_clf,
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearSVC(
                C=0.1, max_iter=5000, random_state=42,
                class_weight="balanced")),
        ]),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=5, gamma=1.0,
            reg_alpha=2.0, reg_lambda=5.0,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, **xgb_gpu),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            num_leaves=7, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=3, reg_alpha=2.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbose=-1, **lgbm_gpu),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, max_depth=2, learning_rate=0.05,
            min_samples_leaf=3, l2_regularization=5.0, random_state=42),
    }
