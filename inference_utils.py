# inference_utils.py
import numpy as np
import pandas as pd
import joblib
import warnings
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

warnings.filterwarnings("ignore")

# --- Базовые математические функции предобработки ---
def fast_baseline(y, degree=6):
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(7):
        c   = np.polyfit(x, y, degree, w=w)
        bl  = np.polyval(c, x)
        res = y - bl
        thr = np.percentile(res, 15)
        rng = max(res.max() - thr, 1e-10)
        w   = np.where(res <= thr, 1.0, np.clip(1.0 - (res - thr) / rng, 0.05, 1.0))
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
    elif norm == "area":
        a = np.trapz(np.abs(s))
        if a > 1e-10: s /= a
    return s

def get_raman_bands(grid):
    w_min, w_max = grid.min(), grid.max()
    center = (w_min + w_max) / 2
    if center < 2000:
        bands =[(900, 1000, "C-C_skel"), (1000, 1100, "phe"), (1100, 1200, "CN_str"),
                 (1200, 1350, "amide_III"), (1350, 1500, "CH2_def"), (1580, 1640, "CC_lip"),
                 (1640, 1700, "amide_I")]
    else:
        bands =[(2700, 2800, "CH_ov"), (2820, 2870, "CH2_sym"), (2870, 2920, "CH3_sym"),
                 (2920, 2970, "CH2_asy"), (2970, 3020, "CH3_asy"), (3020, 3100, "CH_str")]
    return[(lo, hi, nm) for lo, hi, nm in bands if lo >= w_min - 5 and hi <= w_max + 5]

def extract_map_features(pixels_proc, grid, bands):
    feats =[]
    for lo, hi, band_name in bands:
        mask = (grid >= lo) & (grid <= hi)
        if mask.sum() == 0:
            feats.extend([0.0] * 10)
            continue
        seg = pixels_proc[:, mask]
        pix_means = seg.mean(axis=1)
        pix_areas = np.trapz(seg, grid[mask], axis=1)
        median, p25, p75 = np.median(pix_means), np.percentile(pix_means, 25), np.percentile(pix_means, 75)
        std, maxi, iqr = pix_means.std(), pix_means.max(), p75 - p25
        centered = pix_means - median
        skew = (centered**3).mean() / (std**3 + 1e-10)
        kurt = (centered**4).mean() / (std**4 + 1e-10) - 3
        feats.extend([median, p25, p75, std, maxi, iqr, skew, kurt, np.median(pix_areas), pix_areas.std()])
    return feats

def extract_ratio_features(pixels_proc, grid, bands):
    band_medians =[]
    for lo, hi, _ in bands:
        mask = (grid >= lo) & (grid <= hi)
        band_medians.append(np.median(pixels_proc[:, mask].mean(axis=1)) if mask.sum() > 0 else 0.0)
    feats =[]
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            feats.append(band_medians[i] / (band_medians[j] + 1e-10))
    return feats

# --- Класс-обертка для Бэкендера ---
class RamanPredictor:
    def __init__(self, model_pkl_path):
        payload = joblib.load(model_pkl_path)
        self.model = payload["model"]
        self.le = payload["label_encoder"]
        self.region_enc = payload["region_encoder"]
        self.grid = payload["grid"]
        self.norm = payload["norm"]
        self.use_als = payload["use_als"]
        self.wave_min, self.wave_max = payload["wave_range"]

    def _load_raw_file(self, filepath):
        df = pd.read_csv(filepath, sep=r"\s+", comment="#", names=["X", "Y", "Wave", "Intensity"], dtype=np.float64).dropna()
        if self.wave_min: df = df[df["Wave"] >= self.wave_min]
        if self.wave_max: df = df[df["Wave"] <= self.wave_max]
        
        raw_spectra = []
        for _, pix in df.groupby(["X", "Y"], sort=False):
            pix = pix.sort_values("Wave")
            if len(pix) >= 20:
                raw_spectra.append((pix["Wave"].values, pix["Intensity"].values))
        return raw_spectra

    def predict(self, filepath, region_name="unknown"):
        # 1. Загрузка файла
        raw_spectra = self._load_raw_file(filepath)
        if not raw_spectra:
            raise ValueError("Не удалось загрузить данные или файл пуст.")

        # 2. Интерполяция на сетку, на которой училась модель!
        pixels = np.array([np.interp(self.grid, s[0], s[1]) for s in raw_spectra])

        # 3. Предобработка
        pix_proc = np.array([preprocess_spectrum(px, self.grid, self.use_als, self.norm) for px in pixels])

        # 4. Извлечение признаков
        bands = get_raman_bands(self.grid)
        f1 = extract_map_features(pix_proc, self.grid, bands)
        f2 = extract_ratio_features(pix_proc, self.grid, bands)
        features = np.concatenate([f1, f2]).reshape(1, -1)

        # 5. Добавление региона
        # Ожидаемые регионы: "cortex", "striatum", "cerebellum", "unknown"
        reg_encoded = self.region_enc.transform([[region_name]])
        final_X = np.hstack([features, reg_encoded])
        final_X = np.nan_to_num(final_X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # 6. Предсказание
        pred_idx = self.model.predict(final_X)[0]
        pred_label = self.le.inverse_transform([pred_idx])[0]
        
        probas = self.model.predict_proba(final_X)[0]
        proba_dict = {self.le.inverse_transform([i])[0]: float(probas[i]) for i in range(len(probas))}

        return {
            "prediction": pred_label,
            "probabilities": proba_dict,
            "region_used": region_name
        }