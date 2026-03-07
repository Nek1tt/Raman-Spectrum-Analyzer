# 🔬 Raman Spectra Classification — HSP70 Expression Detection

Решение задачи классификации рамановских спектров мозговой ткани для определения экспериментальной группы: **контроль**, **эндогенная экспрессия HSP70**, **экзогенная экспрессия HSP70**.

---

## 📁 Структура репозитория

```
Roman_spectre/
├── baseline_improved.py       # Основной скрипт обучения и оценки моделей
├── requirements.txt           # Зависимости Python
├── README.md                  # Документация
│
├── control/                   # Данные контрольной группы
│   ├── mk1/                   # 12 карт (control_1)
│   ├── mk2a/                  # 24 карты (control_2a)
│   ├── mk2b/                  # 24 карты (control_2b)
│   └── mk3/                   # 20 карт (control_3)
│
├── endo/                      # Данные эндогенной экспрессии HSP70
│   ├── mend1/                 # 13 карт (endo_1)
│   ├── mend2a/                # 26 карт (endo_2a)
│   ├── mend2b/                # 24 карты (endo_2b)
│   └── mend3/                 # 12 карт (endo_3)
│
├── exo/                       # Данные экзогенной экспрессии HSP70
│   ├── mexo1/                 # 12 карт (exo_1)
│   ├── mexo2a/                # 24 карты (exo_2a)
│   ├── mexo2b/                # 24 карты (exo_2b)
│   └── mexo3/                 # 24 карты (exo_3)
│
└── outputs/                   # Результаты обучения (создаётся автоматически)
    ├── best_model_*.pkl        # Сохранённые ML-модели
    ├── cnn_weights_*.pt        # Веса CNN
    ├── cnn_meta_*.pkl          # Метаданные CNN
    └── *.png                  # Графики и визуализации
```

---

## 🧬 Описание задачи

- **Тип задачи**: Многоклассовая классификация (3 класса)
- **Входные данные**: Рамановские спектры мозговой ткани (два спектральных диапазона: ~1500 и ~2900 см⁻¹)
- **Классы**: `control`, `endo`, `exo`
- **Размер датасета**: 118 карт, ~525 пикселей/карта (~61 950 спектров)

---

## ⚙️ Установка

### 1. Клонирование репозитория

```bash
git clone <repository_url>
cd Roman_spectre
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

> ⚡ Для использования GPU (NVIDIA CUDA) убедитесь, что у вас установлен подходящий драйвер и версия CUDA. Установите нужную версию PyTorch с [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🚀 Запуск

### Базовый запуск

```bash
python baseline_improved.py --data_root ./
```

### Полный запуск с оптимальными параметрами

```bash
python baseline_improved.py \
    --data_root ./ \
    --n_grid 512 \
    --norm snv \
    --use_als \
    --fuse_bands \
    --save_plots \
    --optuna_trials_ridge 50 \
    --optuna_trials_cnn 25 \
    --optuna_cnn_epochs 15 \
    --cnn_epochs 120 \
    --cnn_batch 256 \
    --cnn_patience 30 \
    --n_jobs -1
```

### Windows (одна строка)

```cmd
python baseline_improved.py --data_root ./ --n_grid 512 --norm snv --use_als --fuse_bands --save_plots --optuna_trials_ridge 50 --optuna_trials_cnn 25 --optuna_cnn_epochs 15 --cnn_epochs 120 --cnn_batch 256 --cnn_patience 30 --n_jobs -1
```

---

## 🔧 Параметры командной строки

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--data_root` | `./` | Путь к корневой папке с данными |
| `--n_grid` | `512` | Размер интерполяционной сетки по оси волновых чисел |
| `--norm` | `snv` | Метод нормировки: `snv`, `peak_phe`, `area`, `minmax` |
| `--use_als` | `False` | Коррекция базовой линии методом ALS |
| `--fuse_bands` | `False` | Объединение двух спектральных диапазонов в двухканальное представление |
| `--save_plots` | `False` | Сохранять графики (PCA, матрицы ошибок, saliency-карты) |
| `--n_jobs` | `1` | Число параллельных потоков (`-1` = все ядра) |
| `--use_gpu` | `False` | Принудительно использовать GPU |
| `--force_cpu` | `False` | Принудительно использовать CPU |
| `--skip_cnn` | `False` | Пропустить обучение CNN |
| `--skip_ml` | `False` | Пропустить обучение ML-моделей |
| `--cnn_epochs` | `100` | Максимальное число эпох CNN |
| `--cnn_batch` | `128` | Размер батча CNN |
| `--cnn_lr` | `0.001` | Начальный learning rate |
| `--cnn_weight_decay` | `1e-4` | L2-регуляризация |
| `--cnn_dropout` | `0.3` | Вероятность dropout |
| `--cnn_patience` | `20` | Patience для ранней остановки |
| `--optuna_trials_ridge` | `30` | Число испытаний Optuna для Ridge |
| `--optuna_trials_cnn` | `15` | Число испытаний Optuna для CNN |
| `--optuna_cnn_epochs` | `10` | Эпохи CNN в рамках поиска Optuna |
| `--load_model` | — | Путь к ранее обученной ML-модели (`.pkl`) для предсказания |
| `--load_cnn` | — | Путь к весам CNN (`.pt`) для предсказания |
| `--cnn_meta` | — | Путь к метаданным CNN (`.pkl`) |
| `--predict_dir` | — | Директория с новыми данными для инференса |
| `--permutation_test` | `False` | Запустить тест перестановок |
| `--n_permutations` | `100` | Число перестановок |

---

## 🏗️ Архитектура пайплайна

```
Сырые спектры (.txt / .csv)
        │
        ▼
Интерполяция на единую сетку (n_grid=512)
        │
        ▼
Коррекция базовой линии (ALS, опционально)
        │
        ▼
Нормировка (SNV / Peak / Area / MinMax)
        │
        ▼
Извлечение признаков (6 диапазонов × {mean, std, max, skew, kurt, ...} = 75 признаков)
        │
   ┌────┴────┐
   ▼         ▼
 ML-модели  CNN (2-канальный 1D ResNet + SE-блок)
(Ridge,      │
 SVM,        │
 XGB,        │
 LGBM)       │
   └────┬────┘
        ▼
 Оценка: LOGO / SGKF / GSS
        │
        ▼
    Результаты + визуализации
```

---

## 📊 Стратегии кросс-валидации

| Стратегия | Обозначение | Описание |
|-----------|-------------|----------|
| Leave-One-Group-Out | **LOGO** | Оставляет одно животное в тесте. Наиболее строгая оценка обобщающей способности. |
| StratifiedGroupKFold | **SGKF** | 4-фолдная CV с сохранением баланса классов и группировкой по животным. |
| GroupShuffleSplit | **GSS** | Случайное разбиение 10 раз с разделением по животным. |

---

## 📈 Результаты (LOGO accuracy)

### center1500 (низкочастотный диапазон)

| Модель | LOGO acc | SGKF acc | GSS acc |
|--------|----------|----------|---------|
| **RidgeClf** | **0.377** | **0.441** | 0.362 |
| LogReg | 0.377 | 0.434 | 0.360 |
| CNN-1D-ResNet | 0.356 | — | — |
| LinearSVC | 0.341 | 0.433 | 0.327 |
| HistGB | 0.291 | 0.408 | 0.271 |
| LightGBM | 0.283 | 0.407 | 0.268 |
| XGBoost | 0.279 | 0.406 | 0.265 |

### center2900 (высокочастотный диапазон)

| Модель | LOGO acc | SGKF acc | GSS acc |
|--------|----------|----------|---------|
| **RidgeClf** | **0.426** | **0.436** | 0.376 |
| LogReg | 0.409 | 0.423 | 0.375 |
| CNN-1D-ResNet | ~0.371 | — | — |
| LinearSVC | 0.365 | 0.421 | 0.310 |
| HistGB | 0.301 | 0.408 | 0.277 |
| LightGBM | 0.298 | 0.410 | 0.275 |
| XGBoost | 0.298 | 0.412 | 0.272 |

---

## 💾 Инференс на новых данных

```bash
python baseline_improved.py \
    --data_root ./ \
    --load_model outputs/best_model_center1500.pkl \
    --load_cnn outputs/cnn_weights_center1500.pt \
    --cnn_meta outputs/cnn_meta_center1500.pkl \
    --predict_dir ./new_data/
```

---

## 🖥️ Требования к оборудованию

| Компонент | Рекомендуется | Минимум |
|-----------|--------------|---------|
| CPU | 8+ ядер | 4 ядра |
| RAM | 32 GB | 16 GB |
| GPU | NVIDIA RTX 3060+ (CUDA) | — (CPU режим) |
| Диск | 10 GB свободно | 5 GB |

> Полный запуск (center1500 + center2900, CNN + ML) занял ~7 часов на RTX 4060 Ti.

---

## 📦 Выходные файлы

После обучения в папке `outputs/` появятся:

- `best_model_center1500.pkl` / `best_model_center2900.pkl` — лучшие ML-модели
- `cnn_weights_center1500.pt` / `cnn_weights_center2900.pt` — веса CNN
- `cnn_meta_center1500.pkl` / `cnn_meta_center2900.pkl` — метаданные CNN
- `pca_center1500.png` / `pca_center2900.png` — PCA-визуализация признаков
- `cv_all_center1500.png` / `cv_all_center2900.png` — сравнение моделей по CV
- `cm_LOGO_RidgeClf_center*.png` — матрица ошибок лучшей модели
- `saliency_CNN_Saliency_center*.png` — saliency-карты CNN

---

## 🔬 Описание данных

Спектры получены на конфокальном рамановском микроскопе **Renishaw inVia Qontor**.

- **Два спектральных окна**:
  - `center1500`: область ~927–2002 см⁻¹ (белки, липиды, нуклеиновые кислоты)
  - `center2900`: область ~2650–3288 см⁻¹ (CH-колебания, липиды)
- **Разметка**: 3 класса по группам животных (контроль, эндо, экзо HSP70)
- **Организация**: 12 животных (групп), каждая карта содержит ~525 пикселей

---

## 🤝 Авторы

Разработано в рамках хакатона по анализу рамановских спектров биологических тканей.
