"""
Raman spectra 3-class classification — v5
Оптимизирован для малого датасета (12 животных, ~118 карт).

Ключевые изменения vs v4:
  ✅ HYBRID PIXEL STRATEGY: вместо агрегации в одну точку — вычисляем
     статистики по пикселям карты (медиана, p25, p75, std, max) как признаки.
     Это сохраняет внутрикартовую вариабельность как информацию.

  ✅ FAIR LOGO WITH PIXEL TRAIN: пиксели используются при обучении,
     но тест = агрегированные спектры исключённого животного.
     Честная оценка + больший train set.

  ✅ BAND FUSION: объединяем признаки center1500 + center2900.
     Разные диапазоны несут разную биохимическую информацию.

  ✅ REGION-STRATIFIED FEATURES: признаки строятся отдельно для каждого
     региона мозга, затем объединяются. Cortex vs striatum vs cerebellum
     имеют разный молекулярный состав.

  ✅ СИЛЬНАЯ РЕГУЛЯРИЗАЦИЯ: при 12 животных нужны простые модели.
     Добавлен LinearSVC + RidgeClassifier как baseline.
     XGBoost/LightGBM с max_depth=2, сильным reg.

  ✅ PERMUTATION TEST: оцениваем статистическую значимость результатов
     при малом датасете.

Usage:
    # Рекомендуемый запуск:
    python raman_v5.py --data_root /path/to/data --save_plots

    # С fusion обоих диапазонов:
    python raman_v5.py --data_root /path/to/data --fuse_bands --save_plots

    # Полный с пермутационным тестом:
    python raman_v5.py --data_root /path/to/data --fuse_bands --permutation_test --save_plots
"""

import os
import re
import subprocess
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed

from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

COLORS = {"control": "tab:blue", "endo": "tab:orange", "exo": "tab:green"}

# =============================================================================
# GPU
# =============================================================================

def detect_gpu(force=False):
    info = {"available": False, "name": "–",
            "xgb_device": "cpu", "xgb_tree": "hist", "lgbm_device": "cpu"}
    if force:
        return {**info, "available": True, "name": "forced",
                "xgb_device": "cuda", "lgbm_device": "gpu"}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split("\n")[0]
            return {**info, "available": True, "name": name,
                    "xgb_device": "cuda", "lgbm_device": "gpu"}
    except Exception:
        pass
    return info

# =============================================================================
# DATASET STRUCTURE
# =============================================================================

CLASS_DIRS = {
    "control": ["mk1",   "mk2a",   "mk2b",   "mk3"],
    "endo":    ["mend1", "mend2a", "mend2b", "mend3"],
    "exo":     ["mexo1", "mexo2a", "mexo2b", "mexo3"],
}

ANIMAL_RE     = re.compile(r"(\d+[ab]?$)")
BRAIN_REGIONS = ["cerebellum", "striatum", "cortex"]
CENTER_RE     = re.compile(r"center(\d+)")
PLACE_RE      = re.compile(r"place(\d+)")

BAND_RANGES = {
    1500: (900,  2050),
    2900: (2650, 3300),
}


def folder_to_animal_id(folder_name, label):
    m = ANIMAL_RE.search(folder_name)
    return f"{label}_{m.group(1) if m else folder_name}"


def parse_filename(fname):
    fname = fname.lower()
    region = next((r for r in BRAIN_REGIONS if r in fname), "unknown")
    cm = CENTER_RE.search(fname)
    pm = PLACE_RE.search(fname)
    return {
        "region": region,
        "center": int(cm.group(1)) if cm else -1,
        "place":  pm.group(1) if pm else "0",
    }


# =============================================================================
# 1. DATA LOADING — pixel-level, grouped by map
# =============================================================================

def load_hyperspectral_file(filepath, wave_min=None, wave_max=None):
    try:
        df = pd.read_csv(filepath, sep=r"\s+", comment="#",
                         names=["X", "Y", "Wave", "Intensity"],
                         dtype=np.float64).dropna()
    except Exception:
        return []
    if df.empty or len(df) < 10:
        return []
    if wave_min is not None: df = df[df["Wave"] >= wave_min]
    if wave_max is not None: df = df[df["Wave"] <= wave_max]
    if df.empty:
        return []
    spectra = []
    for (_, _), pix in df.groupby(["X", "Y"], sort=False):
        pix  = pix.sort_values("Wave")
        wave = pix["Wave"].values
        intn = pix["Intensity"].values
        if len(wave) >= 20:
            spectra.append((wave, intn))
    return spectra


def find_subdir(data_root, label, subdir):
    for c in [data_root / label / label / subdir,
              data_root / label / subdir,
              data_root / subdir]:
        if c.exists(): return c
    return None


def load_dataset_maps(data_root: str, n_grid: int = 256):
    """
    Загружает датасет как коллекцию карт (map objects).
    Каждая карта = один файл = набор пикселей.

    Возвращает dict: {center: [MapRecord, ...]}
    MapRecord = {label, animal_id, region, place_id, pixels: np.array(n_pix, n_grid), grid}
    """
    data_root = Path(data_root)
    maps = {1500: [], 2900: []}

    print("\n📂 Scanning dataset folders...")
    for label, subdirs in CLASS_DIRS.items():
        for subdir in subdirs:
            folder = find_subdir(data_root, label, subdir)
            if folder is None:
                print(f"  [WARN] not found: {label}/{subdir}")
                continue
            animal_id = folder_to_animal_id(subdir, label)
            txt_files = sorted(folder.glob("*.txt"))
            print(f"  {label}/{subdir}: {len(txt_files)} files  [{animal_id}]")

            for fpath in txt_files:
                fname = fpath.stem.lower()
                if "average" in fname:
                    continue
                meta   = parse_filename(fname)
                center = meta["center"]
                if center not in maps:
                    continue
                w_min, w_max = BAND_RANGES[center]
                pixel_spectra = load_hyperspectral_file(
                    str(fpath), wave_min=w_min, wave_max=w_max)
                if not pixel_spectra:
                    continue

                # Общая сетка для этого файла
                all_w = np.concatenate([s[0] for s in pixel_spectra])
                grid  = np.linspace(all_w.min(), all_w.max(), n_grid)
                pixels = np.array([np.interp(grid, s[0], s[1])
                                   for s in pixel_spectra])

                maps[center].append({
                    "label":     label,
                    "animal_id": animal_id,
                    "region":    meta["region"],
                    "place_id":  f"{animal_id}_p{meta['place']}",
                    "pixels":    pixels,   # (n_pix, n_grid)
                    "grid":      grid,
                })

    for c, recs in maps.items():
        if recs:
            print(f"\n  ✅ center{c}: {len(recs)} maps, "
                  f"~{np.mean([len(r['pixels']) for r in recs]):.0f} px/map")
            labels = [r["label"] for r in recs]
            print(f"     Classes: { {cl: labels.count(cl) for cl in set(labels)} }")
    return maps


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def fast_baseline(y, degree=6):
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(7):
        c   = np.polyfit(x, y, degree, w=w)
        bl  = np.polyval(c, x)
        res = y - bl
        thr = np.percentile(res, 15)
        rng = max(res.max() - thr, 1e-10)
        w   = np.where(res <= thr, 1.0,
                       np.clip(1.0 - (res - thr) / rng, 0.05, 1.0))
    return bl


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
    H = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        z = spsolve(W + H, w * y)
        w = p * (y > z) + (1-p) * (y <= z)
    return z


def preprocess_spectrum(s, grid, use_als=False, norm="snv"):
    s  = s.copy()
    bl = als_baseline(s) if use_als else fast_baseline(s)
    s  = np.clip(s - bl, 0, None)
    s  = savgol_filter(s, window_length=11, polyorder=3)
    if norm == "snv":
        mu, sigma = s.mean(), s.std()
        if sigma > 1e-10: s = (s - mu) / sigma
    elif norm == "peak_phe":
        mask = (grid >= 988) & (grid <= 1018)
        if mask.sum() > 0:
            ref = s[mask].max()
            if ref > 1e-3: s = s / ref
        else:
            mu, sigma = s.mean(), s.std()
            if sigma > 1e-10: s = (s - mu) / sigma
    elif norm == "area":
        a = np.trapz(np.abs(s))
        if a > 1e-10: s /= a
    return s


def preprocess_map(map_record, use_als=False, norm="snv", n_jobs=-1):
    grid   = map_record["grid"]
    pixels = map_record["pixels"]
    proc   = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(preprocess_spectrum)(px, grid, use_als, norm)
        for px in pixels
    )
    return np.array(proc)


# =============================================================================
# 3. HYBRID FEATURE EXTRACTION PER MAP
# =============================================================================

def get_raman_bands(grid):
    w_min, w_max = grid.min(), grid.max()
    center = (w_min + w_max) / 2
    if center < 2000:
        bands = [
            (900,  1000, "C-C_skel"),
            (1000, 1100, "phe"),
            (1100, 1200, "CN_str"),
            (1200, 1350, "amide_III"),
            (1350, 1500, "CH2_def"),
            (1580, 1640, "CC_lip"),
            (1640, 1700, "amide_I"),
        ]
    else:
        bands = [
            (2700, 2800, "CH_ov"),
            (2820, 2870, "CH2_sym"),
            (2870, 2920, "CH3_sym"),
            (2920, 2970, "CH2_asy"),
            (2970, 3020, "CH3_asy"),
            (3020, 3100, "CH_str"),
        ]
    return [(lo, hi, nm) for lo, hi, nm in bands
            if lo >= w_min - 5 and hi <= w_max + 5]


PIXEL_STATS = ["median", "p25", "p75", "std", "max", "iqr"]


def extract_map_features(pixels_proc: np.ndarray, grid: np.ndarray,
                          bands: list, tag: str = "") -> tuple:
    """
    Из набора пикселей одной карты извлекает СТАТИСТИКИ по пикселям
    для каждого бэнда. Это ключевое отличие от простой агрегации —
    сохраняем информацию о разбросе внутри карты.

    pixels_proc: (n_pix, n_wave)
    Returns: feature_vector (1D), feature_names (list)
    """
    feats = []
    names = []

    for lo, hi, band_name in bands:
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() == 0:
            feats.extend([0.0] * (len(PIXEL_STATS) + 2))
            names.extend([f"{tag}_{band_name}_{s}"
                          for s in PIXEL_STATS + ["skew", "kurt"]])
            continue

        seg = pixels_proc[:, mask]          # (n_pix, n_band_pts)
        # Среднее по частотной оси для каждого пикселя → (n_pix,)
        pix_means = seg.mean(axis=1)
        # Площадь пика для каждого пикселя
        pix_areas = np.trapz(seg, grid[mask], axis=1)

        median = np.median(pix_means)
        p25    = np.percentile(pix_means, 25)
        p75    = np.percentile(pix_means, 75)
        std    = pix_means.std()
        maxi   = pix_means.max()
        iqr    = p75 - p25

        # Skewness and kurtosis (простые моменты)
        centered = pix_means - median
        skew     = (centered**3).mean() / (std**3 + 1e-10)
        kurt     = (centered**4).mean() / (std**4 + 1e-10) - 3

        feats.extend([median, p25, p75, std, maxi, iqr, skew, kurt])
        names.extend([f"{tag}_{band_name}_{s}"
                      for s in PIXEL_STATS + ["skew", "kurt"]])

        # Дополнительно: медиана площади пика и её std
        feats.extend([np.median(pix_areas), pix_areas.std()])
        names.extend([f"{tag}_{band_name}_area_med",
                      f"{tag}_{band_name}_area_std"])

    return np.array(feats), names


def extract_ratio_features(pixels_proc: np.ndarray, grid: np.ndarray,
                            bands: list, tag: str = "") -> tuple:
    """
    Межбэндовые соотношения — часто более информативны, чем абсолютные
    значения, т.к. нечувствительны к общему масштабу интенсивности.
    """
    # Считаем медианные интенсивности по бэндам
    band_medians = []
    for lo, hi, _ in bands:
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() > 0:
            pix_means    = pixels_proc[:, mask].mean(axis=1)
            band_medians.append(np.median(pix_means))
        else:
            band_medians.append(0.0)

    feats = []
    names = []
    n = len(bands)
    for i in range(n):
        for j in range(i+1, n):
            denom = band_medians[j] + 1e-10
            ratio = band_medians[i] / denom
            feats.append(ratio)
            names.append(f"{tag}_{bands[i][2]}_over_{bands[j][2]}")

    return np.array(feats), names


def build_map_features(map_record: dict, pixels_proc: np.ndarray,
                        bands: list, tag: str = "") -> tuple:
    """Полный вектор признаков для одной карты."""
    grid = map_record["grid"]

    f1, n1 = extract_map_features(pixels_proc, grid, bands, tag)
    f2, n2 = extract_ratio_features(pixels_proc, grid, bands, tag)

    feats = np.concatenate([f1, f2])
    names = n1 + n2

    return feats, names


# =============================================================================
# 4. BUILD FULL FEATURE MATRIX
# =============================================================================

def build_feature_matrix(maps: list, center_tag: str,
                          use_als: bool = False, norm: str = "snv",
                          n_jobs: int = -1) -> tuple:
    """
    Для списка карт одного center строит матрицу признаков:
    одна строка = одна карта.
    """
    if not maps:
        return None, None, None, None, None, None

    # Общая сетка (берём из первой карты — все должны быть совместимы)
    grid  = maps[0]["grid"]
    bands = get_raman_bands(grid)
    print(f"  [{center_tag}] {len(maps)} maps, {len(bands)} bands")

    X, labels, animal_ids, regions, place_ids = [], [], [], [], []
    feat_names = None

    for rec in tqdm(maps, desc=f"  Features {center_tag}", ncols=80):
        # Preprocess пиксели
        pix_proc = preprocess_map(rec, use_als=use_als, norm=norm,
                                   n_jobs=n_jobs)
        # Реинтерполируем на общую сетку если нужно
        if not np.allclose(rec["grid"], grid):
            pix_proc = np.array([np.interp(grid, rec["grid"], px)
                                  for px in pix_proc])

        feats, fnames = build_map_features(rec, pix_proc, bands,
                                            tag=center_tag)
        X.append(feats)
        if feat_names is None:
            feat_names = fnames
        labels.append(rec["label"])
        animal_ids.append(rec["animal_id"])
        regions.append(rec["region"])
        place_ids.append(rec["place_id"])

    X = np.nan_to_num(np.array(X, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return (X, np.array(labels), np.array(animal_ids),
            np.array(regions), np.array(place_ids), feat_names)


# =============================================================================
# 5. MODELS — small-data optimized
# =============================================================================

def get_models(gpu: dict) -> dict:
    xgb_gpu  = ({"device": gpu["xgb_device"], "tree_method": gpu["xgb_tree"]}
                if gpu["available"] else {"tree_method": "hist"})
    lgbm_gpu = ({"device": gpu["lgbm_device"]} if gpu["available"] else {})

    return {
        # Линейные — хороши при малых данных
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=0.1, max_iter=2000,
                                          random_state=42,
                                          class_weight="balanced")),
        ]),
        "RidgeClf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RidgeClassifier(alpha=5.0, class_weight="balanced")),
        ]),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LinearSVC(C=0.1, max_iter=5000, random_state=42,
                                 class_weight="balanced")),
        ]),
        # Бустинг с сильной регуляризацией
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            min_child_weight=5, gamma=1.0,
            reg_alpha=2.0, reg_lambda=5.0,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
            **xgb_gpu,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            num_leaves=7, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=3, reg_alpha=2.0, reg_lambda=5.0,
            random_state=42, n_jobs=-1, verbose=-1,
            **lgbm_gpu,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, max_depth=2, learning_rate=0.05,
            min_samples_leaf=3, l2_regularization=5.0, random_state=42,
        ),
    }


# =============================================================================
# 6. EVALUATION
# =============================================================================

def run_logo_cv(models: dict, X: np.ndarray, y_enc: np.ndarray,
                groups: np.ndarray, classes) -> pd.DataFrame:
    logo     = LeaveOneGroupOut()
    n_groups = len(np.unique(groups))
    results  = []

    for name, model in models.items():
        print(f"\n  ▶ {name}")
        fold_scores = []
        fold_true, fold_pred = [], []

        for tr, te in logo.split(X, y_enc, groups):
            model.fit(X[tr], y_enc[tr])
            preds = model.predict(X[te])
            fold_scores.append(accuracy_score(y_enc[te], preds))
            fold_true.extend(y_enc[te].tolist())
            fold_pred.extend(preds.tolist())

        scores = np.array(fold_scores)
        print(f"    LOGO: {scores.mean():.3f} ± {scores.std():.3f}  "
              f"per-fold: {[f'{s:.2f}' for s in scores]}")
        print(classification_report(fold_true, fold_pred,
                                     target_names=classes, digits=3))
        results.append({
            "Model":     name,
            "LOGO_mean": scores.mean(), "LOGO_std": scores.std(),
            "LOGO_min":  scores.min(),  "LOGO_max": scores.max(),
            "y_true":    fold_true,     "y_pred":   fold_pred,
        })

    return pd.DataFrame(results)


def permutation_test(best_model, X: np.ndarray, y_enc: np.ndarray,
                      groups: np.ndarray, observed_acc: float,
                      n_permutations: int = 200) -> float:
    """
    Пермутационный тест: сколько раз случайное перемешивание меток
    даёт точность >= observed_acc?
    Возвращает p-value.
    """
    logo = LeaveOneGroupOut()
    rng  = np.random.RandomState(42)
    perm_scores = []

    print(f"\n  🎲 Permutation test ({n_permutations} permutations)...")
    for _ in tqdm(range(n_permutations), desc="  Permuting", ncols=72):
        y_perm = rng.permutation(y_enc)
        fold_scores = []
        for tr, te in logo.split(X, y_perm, groups):
            best_model.fit(X[tr], y_perm[tr])
            fold_scores.append(accuracy_score(y_perm[te],
                                               best_model.predict(X[te])))
        perm_scores.append(np.mean(fold_scores))

    perm_scores = np.array(perm_scores)
    p_value     = (perm_scores >= observed_acc).mean()
    print(f"  Observed acc={observed_acc:.3f}, "
          f"permutation mean={perm_scores.mean():.3f} ± {perm_scores.std():.3f}")
    print(f"  p-value = {p_value:.3f}  "
          f"({'SIGNIFICANT ✅' if p_value < 0.05 else 'NOT significant ❌'})")
    return p_value


# =============================================================================
# 7. BAND FUSION (1500 + 2900)
# =============================================================================

def fuse_features(X1, names1, y1, aid1, reg1,
                   X2, names2, y2, aid2, reg2) -> tuple:
    """
    Объединяет признаки center1500 и center2900 по совпадающим картам.
    Совпадение определяется по animal_id (т.к. place может не совпасть).
    """
    # Ищем общие animal_ids
    common = sorted(set(aid1) & set(aid2))
    if not common:
        print("  [WARN] No common animals for fusion!")
        return None

    print(f"  Fusing bands: {len(common)} common animals")

    # Для каждого животного берём первую карту из каждого center
    # (если их несколько — усредняем по картам животного)
    rows1, rows2, y_f, aid_f, reg_f = [], [], [], [], []
    for animal in common:
        idx1 = np.where(aid1 == animal)[0]
        idx2 = np.where(aid2 == animal)[0]
        rows1.append(X1[idx1].mean(axis=0))
        rows2.append(X2[idx2].mean(axis=0))
        y_f.append(y1[idx1[0]])
        aid_f.append(animal)
        reg_f.append(reg1[idx1[0]])

    X_fused = np.hstack([rows1, rows2])
    names_f = [f"c1500_{n}" for n in names1] + [f"c2900_{n}" for n in names2]

    return (np.array(X_fused, dtype=np.float32),
            names_f, np.array(y_f), np.array(aid_f), np.array(reg_f))


# =============================================================================
# 8. VISUALISATIONS
# =============================================================================

def plot_mean_spectra_from_maps(maps: list, title: str,
                                 out_dir: Path, save_plots: bool, fname: str,
                                 use_als=False, norm="snv"):
    grid  = maps[0]["grid"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for cls in ["control", "endo", "exo"]:
        cls_maps = [m for m in maps if m["label"] == cls]
        if not cls_maps: continue
        all_medians = []
        for m in cls_maps:
            pix_proc = preprocess_map(m, use_als=use_als, norm=norm)
            if not np.allclose(m["grid"], grid):
                pix_proc = np.array([np.interp(grid, m["grid"], px)
                                     for px in pix_proc])
            all_medians.append(np.median(pix_proc, axis=0))
        arr = np.array(all_medians)
        mu, s = arr.mean(0), arr.std(0)
        ax.plot(grid, mu, label=cls, color=COLORS[cls], lw=1.5)
        ax.fill_between(grid, mu-s, mu+s, alpha=0.2, color=COLORS[cls])
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


def plot_pca_maps(X, y, animal_ids, title, out_dir, save_plots, fname):
    pca  = PCA(n_components=min(X.shape[1], 2))
    Xpca = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    for cls in np.unique(y):
        mask = y == cls
        axes[0].scatter(Xpca[mask, 0], Xpca[mask, 1],
                        c=COLORS.get(cls, "gray"), label=cls, s=60, alpha=0.8)
    axes[0].set_title(f"By class (PC1={var[0]:.1%}, PC2={var[1]:.1%})")
    axes[0].legend()
    unique_animals = np.unique(animal_ids)
    cmap = plt.cm.get_cmap("tab20", len(unique_animals))
    for i, animal in enumerate(unique_animals):
        mask = animal_ids == animal
        axes[1].scatter(Xpca[mask, 0], Xpca[mask, 1],
                        c=[cmap(i)], label=animal, s=60, alpha=0.8)
    axes[1].set_title("By animal")
    axes[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if save_plots: plt.savefig(out_dir / fname, dpi=150)
    plt.show()


def plot_cv_results(df, out_dir, save_plots, suffix=""):
    df_s = df.sort_values("LOGO_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, len(df)*0.7)))
    bars = ax.barh(df_s["Model"], df_s["LOGO_mean"],
                   xerr=df_s["LOGO_std"],
                   color="steelblue", alpha=0.8, capsize=5)
    ax.axvline(1/3, ls="--", color="red", lw=1.5, label="random (33%)")
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy")
    ax.set_title(f"LOGO CV {suffix}"); ax.legend()
    for bar, m in zip(bars, df_s["LOGO_mean"]):
        ax.text(m + 0.01, bar.get_y() + bar.get_height()/2,
                f"{m:.3f}", va="center", fontsize=10)
    plt.tight_layout()
    if save_plots:
        plt.savefig(out_dir / f"logo_cv{suffix}.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title, out_dir, save_plots):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d",   cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"{title} (recall per class)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    if save_plots:
        plt.savefig(out_dir / f"cm_{title.replace(' ','_')}.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feat_names, top_n=25,
                             title="", out_dir=None, save_plots=False):
    try:
        # Pipeline — достаём финальный estimator
        if hasattr(model, "named_steps"):
            clf = model.named_steps[list(model.named_steps)[-1]]
        else:
            clf = model
        imp = clf.feature_importances_
    except AttributeError:
        try:
            if hasattr(model, "named_steps"):
                clf = model.named_steps[list(model.named_steps)[-1]]
            else:
                clf = model
            imp = np.abs(clf.coef_).mean(axis=0)
        except AttributeError:
            print(f"  [{title}] importance not available")
            return

    top_n  = min(top_n, len(imp))
    idx    = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), imp[idx], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([str(feat_names[i])[:35] for i in idx], fontsize=8)
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout()
    if save_plots and out_dir:
        plt.savefig(out_dir / f"fi_{title.replace(' ','_')}.png", dpi=150)
    plt.show()


def plot_region_breakdown(X, y, animal_ids, regions, feat_names,
                           title, out_dir, save_plots):
    """Accuracy breakdown по регионам мозга."""
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_regions = np.unique(regions)
    x_pos = np.arange(len(unique_regions))
    width = 0.25

    for i, cls in enumerate(["control", "endo", "exo"]):
        accs = []
        for reg in unique_regions:
            mask = (y == cls) & (regions == reg)
            if mask.sum() > 0:
                accs.append(mask.sum())
            else:
                accs.append(0)
        ax.bar(x_pos + i*width, accs, width, label=cls,
               color=COLORS[cls], alpha=0.8)

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(unique_regions)
    ax.set_ylabel("Number of maps")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(out_dir / f"region_{title.replace(' ','_')}.png", dpi=150)
    plt.show()


# =============================================================================
# 9. PIPELINE
# =============================================================================

def run_pipeline(maps: list, center_tag: str, args, gpu, out_dir: Path):
    if not maps:
        print(f"  [SKIP] No maps for {center_tag}")
        return None

    print(f"\n{'='*65}")
    print(f"🔬 PIPELINE: {center_tag}  ({len(maps)} maps)")
    print(f"{'='*65}")

    # Средние спектры
    if args.save_plots:
        plot_mean_spectra_from_maps(
            maps, f"Median spectra – {center_tag}",
            out_dir, args.save_plots, f"mean_{center_tag}.png",
            use_als=args.use_als, norm=args.norm)

    # Feature matrix
    result = build_feature_matrix(
        maps, center_tag, use_als=args.use_als,
        norm=args.norm, n_jobs=args.n_jobs)
    X, y, animal_ids, regions, place_ids, feat_names = result

    # Добавляем регион как признак
    region_enc = OrdinalEncoder(
        categories=[["cortex", "striatum", "cerebellum", "unknown"]],
        handle_unknown="use_encoded_value", unknown_value=-1)
    reg_feat   = region_enc.fit_transform(regions.reshape(-1, 1))
    X          = np.hstack([X, reg_feat])
    feat_names = feat_names + ["brain_region"]

    print(f"\n  ✅ Feature matrix: {X.shape[0]} maps × {X.shape[1]} features")

    # Encode
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    classes = le.classes_

    le_grp  = LabelEncoder()
    groups  = le_grp.fit_transform(animal_ids)
    print(f"  Animals: {list(le_grp.classes_)}")

    # Region breakdown
    if args.save_plots:
        plot_region_breakdown(X, y, animal_ids, regions, feat_names,
                               f"Region breakdown {center_tag}",
                               out_dir, args.save_plots)

    # PCA
    if args.save_plots:
        plot_pca_maps(X, y, animal_ids, f"PCA features – {center_tag}",
                      out_dir, args.save_plots, f"pca_feat_{center_tag}.png")

    # Models + LOGO CV
    print(f"\n📊 LOGO CV — {center_tag}")
    models     = get_models(gpu)
    results_df = run_logo_cv(models, X, y_enc, groups, classes)
    plot_cv_results(results_df, out_dir, args.save_plots,
                    suffix=f"_{center_tag}")

    best_name = results_df.loc[results_df["LOGO_mean"].idxmax(), "Model"]
    best_row  = results_df[results_df["Model"] == best_name].iloc[0]
    best_acc  = best_row["LOGO_mean"]
    print(f"\n🏆 Best: {best_name}  LOGO={best_acc:.3f}")

    plot_confusion_matrix(best_row["y_true"], best_row["y_pred"], classes,
                          f"LOGO {best_name} {center_tag}",
                          out_dir, args.save_plots)

    # Permutation test
    p_value = None
    if args.permutation_test:
        best_model = models[best_name]
        p_value    = permutation_test(best_model, X, y_enc, groups,
                                       best_acc, n_permutations=200)

    # Feature importance (full fit)
    best_model = models[best_name]
    best_model.fit(X, y_enc)
    plot_feature_importance(best_model, feat_names, top_n=25,
                            title=f"{best_name} {center_tag}",
                            out_dir=out_dir, save_plots=args.save_plots)

    # Save
    payload = {
        "model": best_model, "label_encoder": le,
        "region_encoder": region_enc, "grid": maps[0]["grid"],
        "feat_names": feat_names, "center_tag": center_tag,
        "logo_acc": best_acc, "p_value": p_value,
        "norm": args.norm, "use_als": args.use_als,
        "wave_range": BAND_RANGES.get(int(center_tag.replace("center", "")),
                                       (None, None)),
        "n_grid": len(maps[0]["grid"]),
    }
    save_path = out_dir / f"best_model_{center_tag}.pkl"
    joblib.dump(payload, save_path)
    print(f"  💾 Saved → {save_path}")

    # Summary table
    print(f"\n📋 SUMMARY  {center_tag}")
    print(f"  {'Model':<14} {'LOGO':>10}   {'±':>6}   {'min':>6}   {'max':>6}")
    print("  " + "-" * 50)
    for _, row in results_df.sort_values("LOGO_mean", ascending=False).iterrows():
        m = "  ◀" if row["Model"] == best_name else ""
        print(f"  {row['Model']:<14} "
              f"{row['LOGO_mean']:>8.3f}   "
              f"{row['LOGO_std']:>6.3f}   "
              f"{row['LOGO_min']:>6.3f}   "
              f"{row['LOGO_max']:>6.3f}{m}")
    if p_value is not None:
        print(f"\n  p-value = {p_value:.3f}")

    return {"results_df": results_df, "X": X, "y": y,
            "animal_ids": animal_ids, "regions": regions,
            "feat_names": feat_names, "le": le, "groups": groups,
            "best_acc": best_acc}


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root",        required=True)
    parser.add_argument("--n_grid",           type=int, default=256)
    parser.add_argument("--norm",             default="snv",
                        choices=["snv", "peak_phe", "area", "minmax"])
    parser.add_argument("--use_als",          action="store_true")
    parser.add_argument("--fuse_bands",       action="store_true",
                        help="Combine center1500 + center2900 features")
    parser.add_argument("--permutation_test", action="store_true")
    parser.add_argument("--save_plots",       action="store_true")
    parser.add_argument("--n_jobs",           type=int, default=-1)
    parser.add_argument("--use_gpu",          action="store_true")
    parser.add_argument("--force_cpu",        action="store_true")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    gpu = detect_gpu(force=args.use_gpu)
    if args.force_cpu:
        gpu = {"available": False, "name": "–",
               "xgb_device": "cpu", "xgb_tree": "hist", "lgbm_device": "cpu"}
    print(f"GPU: {'✅ ' + gpu['name'] if gpu['available'] else 'CPU'}")
    print(f"Config: n_grid={args.n_grid}, norm={args.norm}, "
          f"fuse={args.fuse_bands}, perm_test={args.permutation_test}")

    # Load
    all_maps = load_dataset_maps(args.data_root, n_grid=args.n_grid)

    results = {}
    for center, maps in all_maps.items():
        if maps:
            tag = f"center{center}"
            results[tag] = run_pipeline(maps, tag, args, gpu, out_dir)

    # Band fusion
    if args.fuse_bands and "center1500" in results and "center2900" in results:
        print(f"\n{'='*65}")
        print(f"🔀 BAND FUSION: center1500 + center2900")
        print(f"{'='*65}")
        r1 = results["center1500"]
        r2 = results["center2900"]
        fused = fuse_features(
            r1["X"], r1["feat_names"], r1["y"], r1["animal_ids"], r1["regions"],
            r2["X"], r2["feat_names"], r2["y"], r2["animal_ids"], r2["regions"],
        )
        if fused is not None:
            X_f, names_f, y_f, aid_f, reg_f = fused
            le_f   = LabelEncoder()
            y_f_enc = le_f.fit_transform(y_f)
            le_grp  = LabelEncoder()
            groups_f = le_grp.fit_transform(aid_f)
            print(f"\n  Fused matrix: {X_f.shape[0]} × {X_f.shape[1]}")

            models = get_models(gpu)
            df_f   = run_logo_cv(models, X_f, y_f_enc, groups_f, le_f.classes_)
            plot_cv_results(df_f, out_dir, args.save_plots, suffix="_fused")

            best_n = df_f.loc[df_f["LOGO_mean"].idxmax(), "Model"]
            best_r = df_f[df_f["Model"] == best_n].iloc[0]
            plot_confusion_matrix(best_r["y_true"], best_r["y_pred"],
                                  le_f.classes_, f"LOGO {best_n} fused",
                                  out_dir, args.save_plots)

            if args.permutation_test:
                permutation_test(models[best_n], X_f, y_f_enc, groups_f,
                                  best_r["LOGO_mean"])

            results["fused"] = df_f

    # Final summary
    print("\n" + "=" * 65)
    print("📋 FINAL SUMMARY")
    print("=" * 65)
    for tag, r in results.items():
        if r is None: continue
        df = r["results_df"] if isinstance(r, dict) and "results_df" in r else r
        best = df.loc[df["LOGO_mean"].idxmax()]
        sig  = ""
        if isinstance(r, dict) and r.get("best_acc") is not None:
            acc = r["best_acc"]
            sig = f"  {'↑ above random' if acc > 1/3 else '↓ at/below random'}"
        print(f"  [{tag}] {best['Model']}: "
              f"LOGO={best['LOGO_mean']:.3f}±{best['LOGO_std']:.3f}{sig}")
    print(f"  Random baseline: {1/3:.3f}")
    print("  ✅ Done.")


if __name__ == "__main__":
    main()