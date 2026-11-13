# AI-Based Bicycle Rental Demand Prediction Using Generalized Additive Models (GAM)

## 1. Project Title
**AI-Based Bicycle Rental Demand Prediction Using Generalized Additive Models (GAM)**

## 2. Problem Statement
Model nonlinear bicycle rental demand relationships with weather and time using the official UCI Bike Sharing Dataset, and assess Generalized Additive Model (GAM) performance against baseline and ensemble machine learning methods.

## 3. Objectives
- Fit smooth temporal terms (`hr`, `mnth`) and smooth weather terms (`temp`, `atemp`, `hum`, `windspeed`) using a Generalized Additive Model (GAM).
- Model categorical temporal and weather factors (`season`, `yr`, `holiday`, `weekday`, `workingday`, `weathersit`).
- Compare GAM performance against Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
- Visualize GAM zero-centered smooth partial dependence curves with 95% confidence intervals.
- Evaluate all models using **MAE**, **RMSE**, and **Poisson Deviance Explained** on a chronological test split.
- Provide an interactive full-stack FastAPI + React demand prediction dashboard powered exclusively by the real dataset and trained models.

## 4. Dataset Description
The **UCI Bike Sharing Dataset** contains hourly and daily counts of rental bikes between 2011 and 2012 in the Capital Bikeshare system (Washington, D.C.) paired with corresponding weather and seasonal information.
- **Primary dataset used:** `data/raw/hour.csv` — **17,379 rows × 17 columns**
- **Daily dataset retained:** `data/raw/day.csv` — **731 rows × 16 columns**
- **Date range:** `2011-01-01` to `2012-12-31`
- **Missing values:** `0`
- **Duplicate rows:** `0`
- **Target (`cnt`):** Total hourly rentals (`min = 1`, `median = 142.0`, `mean = 189.46`, `std = 181.39`, `max = 977`, `skewness = 1.28`).

## 5. Official Dataset Source URL
- **Dataset Page:** https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- **Direct Archive:** https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

## 6. Dataset Download Instructions
```powershell
.\.venv\Scripts\python.exe -m src.data.download_dataset
.\.venv\Scripts\python.exe -m src.data.validate_dataset
```

## 7. Dataset Features
| Feature | Type | Description |
|---|---|---|
| `season` | Categorical (1–4) | 1: Spring, 2: Summer, 3: Fall, 4: Winter |
| `yr` | Categorical (0–1) | Year (0: 2011, 1: 2012) |
| `mnth` | Temporal (1–12) | Month of year |
| `hr` | Temporal (0–23) | Hour of day |
| `holiday` | Binary (0–1) | Whether day is a holiday |
| `weekday` | Categorical (0–6) | Day of week |
| `workingday` | Binary (0–1) | 1 if neither weekend nor holiday |
| `weathersit` | Categorical (1–4) | 1: Clear, 2: Mist/Cloudy, 3: Light Snow/Rain, 4: Heavy Rain/Snow |
| `temp` | Continuous [0, 1] | Normalized temperature in Celsius |
| `atemp` | Continuous [0, 1] | Normalized feeling temperature |
| `hum` | Continuous [0, 1] | Normalized humidity |
| `windspeed` | Continuous [0, 1] | Normalized wind speed |
| `cnt` | Target (Integer) | Total hourly bicycle rental count |

*Excluded to prevent leakage:* `instant` (row ID), `casual`, and `registered` (components summing to `cnt`).

## 8. Project Architecture
- **Data & ML Pipeline (`src/data`, `src/features`, `src/models`, `src/visualization`):** Downloads, validates, chronologically splits, transforms, trains 4 models, evaluates metrics, and generates plots.
- **FastAPI Backend (`src/api`):** Loads persisted `.pkl` models and `feature_config.json` on startup, serving real dataset analytics, GAM partial effects, evaluation metrics, and live predictions.
- **React + Vite + Tailwind Frontend (`frontend/`):** Interactive dashboard consuming FastAPI endpoints via Axios and rendering charts with Recharts.

## 9. Folder Structure
```text
├── data/
│   ├── raw/                  # Official hour.csv, day.csv, Readme.txt
│   └── processed/            # cleaned_hour.csv, train.csv, test.csv
├── models/                   # Persisted .pkl models, scaler.pkl, feature_config.json
├── results/
│   ├── metrics/              # model_comparison.csv
│   ├── predictions/          # test_predictions.csv
│   └── plots/                # EDA, GAM partial effect, and evaluation PNG plots
├── notebooks/                # exploratory_data_analysis.ipynb
├── src/
│   ├── config.py
│   ├── run_pipeline.py
│   ├── data/                 # download_dataset.py, validate_dataset.py, preprocess.py
│   ├── features/             # feature_engineering.py
│   ├── models/               # train_gam.py, train_linear_regression.py, train_random_forest.py, train_gradient_boosting.py, model_evaluation.py
│   ├── visualization/        # eda_visualizations.py, gam_effect_plots.py, evaluation_plots.py
│   └── api/                  # main.py, schemas.py, prediction_service.py
├── frontend/                 # React + Vite + Tailwind + Recharts UI
├── tests/                    # Pytest unit and integration tests
├── requirements.txt
├── pytest.ini
├── run_project.bat
└── README.md
```

## 10. Preprocessing Methodology
- Loaded `data/raw/hour.csv` (17,379 records).
- Removed `instant`, `casual`, and `registered`.
- Verified zero missing values and zero duplicates.
- Enforced integer and float dtypes and sorted chronologically by `(dteday, hr)`.

## 11. Feature Engineering
- Fitted a `ColumnTransformer` (`StandardScaler` for `hr, mnth, temp, atemp, hum, windspeed`; `OneHotEncoder(drop="first")` for `season, yr, holiday, weekday, workingday, weathersit`) **on training data only** and saved to `models/scaler.pkl`.
- Saved dataset-derived bounds and defaults to `models/feature_config.json`.

## 12. GAM Mathematical Explanation
A Generalized Additive Model (GAM) models the expected count $\mu_i = \mathbb{E}[y_i \mid x_i]$ via a link function $g(\cdot)$:

$$g(\mu_i) = \beta_0 + \sum_{j} f_j(x_{ij})$$

Because `cnt` is non-negative, right-skewed count data (`skew = 1.28`), we use a **PoissonGAM** with a **log link** ($g(\mu) = \ln(\mu)$), ensuring non-negative expected counts $\hat{\mu}_i = \exp(\eta_i)$ and modeling multiplicative effects of weather and time.

## 13. GAM Smooth Term Details
- **Smooth spline terms (`s(...)`, cubic B-splines):**
  - `s(hr, n_splines=24)` — captures bimodal commute peaks and nighttime lows
  - `s(temp, n_splines=15)` — captures nonlinear thermal comfort effect
  - `s(atemp, n_splines=15)` — captures feeling temperature nonlinearity
  - `s(hum, n_splines=15)` — captures high-humidity demand drop-off
  - `s(windspeed, n_splines=12)` — captures wind resistance effect
  - `s(mnth, n_splines=12)` — captures seasonal monthly progression
- **Factor terms (`f(...)`):** `season`, `yr`, `holiday`, `weekday`, `workingday`, `weathersit`.
- Smoothing parameter $\lambda$ selected on training data via `gam.gridsearch` using GCV/UBRE.

## 14. Comparison Algorithms
1. **Linear Regression** (`sklearn.linear_model.LinearRegression`)
2. **Random Forest Regressor** (`n_estimators=200`, `max_depth=18`, `min_samples_split=4`, `min_samples_leaf=2`)
3. **Gradient Boosting Regressor** (`n_estimators=300`, `learning_rate=0.08`, `max_depth=5`, `subsample=0.85`, `min_samples_leaf=5`)

## 15. Model Training Methodology
All four models were trained independently on the exact same 80% chronological training split (`13,903` observations) with no access to test data. Predictions are clipped at `0.0` for non-negative rental counts.

## 16. Chronological Train-Test Split
- **Training set (80%):** `13,903` rows (`2011-01-01` to `2012-08-07`)
- **Testing set (20%):** `3,476` rows (`2012-08-07` to `2012-12-31`)

## 17. Evaluation Metrics
- **MAE:** $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **RMSE:** $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **Poisson Deviance Explained:** $1 - \frac{D(y_{\text{test}}, \hat{y}_{\text{test}})}{D(y_{\text{test}}, \bar{y}_{\text{train}})}$, where $D(y, \mu) = 2\sum_i \left(y_i \ln(y_i/\mu_i) - (y_i - \mu_i)\right)$.

## 18. Actual Final Performance Results
Evaluated on the held-out chronological test set (`3,476` observations):

| Model | MAE | RMSE | Poisson Deviance Explained |
|---|---:|---:|---:|
| **Gradient Boosting Regressor** | **46.3860** | **70.7192** | **0.9200** |
| Random Forest Regressor | 49.7899 | 76.1299 | 0.9095 |
| Generalized Additive Model (GAM) | 79.2060 | 113.4433 | 0.8059 |
| Linear Regression | 136.7756 | 182.8389 | 0.4233 |

- **Best-performing model (MAE & RMSE):** **Gradient Boosting Regressor** (`MAE = 46.3860`, `RMSE = 70.7192`).
- **GAM native training deviance explained:** `0.8142` (and `0.8059` on the out-of-sample chronological test set), substantially outperforming Linear Regression (`MAE = 136.7756`, `RMSE = 182.8389`) while providing interpretable smooth partial effect curves.

## 19. Generated Visualizations
Saved in `results/plots/`:
- `demand_distribution.png`
- `hourly_demand.png`
- `weather_demand.png`
- `seasonal_monthly_demand.png`
- `correlation_matrix.png`
- `gam_partial_effects.png`
- `model_comparison.png`
- `actual_vs_predicted.png`
- `residual_plot.png`
- `error_distribution.png`

## 20. API Endpoints
- `GET /` — Health and model/dataset readiness status
- `GET /api/metrics` — Model evaluation metrics and split counts
- `GET /api/model-comparison` — Metrics, recent test time series, scatter points, and residual statistics
- `GET /api/demand-analysis` — Hourly, weather, seasonal, monthly, and temperature demand summaries
- `GET /api/gam-effects` — Zero-centered PoissonGAM smooth partial effect curves and 95% CIs
- `GET /api/feature-info` — Dataset-derived feature bounds and defaults
- `POST /api/predict` — Single-model demand prediction
- `POST /api/predict-all` — Predictions from all four trained models

## 21. Installation Instructions
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
cd frontend
npm install
```

## 22. Execution Instructions
Option A — One-click Windows launcher:
```powershell
.\run_project.bat
```
Option B — Manual execution:
```powershell
.\.venv\Scripts\python.exe -m src.run_pipeline
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
cd frontend && npm run dev
```

## 23. Testing Instructions
```powershell
.\.venv\Scripts\python.exe -m pytest -v
cd frontend && npm run build
```

## 24. Limitations
- Additive GAM terms (`s(hr) + f(workingday)`) do not automatically capture interaction effects such as `hr × workingday` (weekday commute peaks vs weekend midday peaks), which tree-based ensembles (Gradient Boosting, Random Forest) capture natively—explaining their lower test MAE/RMSE.
- The dataset covers 2011–2012 Washington, D.C. ridership; predictions for other cities or years require retraining on local data.

## 25. Future Improvements
- Add tensor-product interaction terms `te(hr, workingday)` or `by=workingday` in `PoissonGAM` to capture distinct weekday/weekend hourly curves.
- Incorporate lag features or rolling weather averages for multi-step forecasting.
