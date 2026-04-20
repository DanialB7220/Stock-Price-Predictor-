# Stock-Price-Predictor-

End-to-end stock trend prediction project in both Python and R, including:
- data pipeline based on local dataset `data/stock data.csv`,
- baseline and advanced ML models,
- dashboards (Dash and Shiny),
- Flask API with `.env` configuration.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional R packages:

```r
install.packages(c("readr", "dplyr", "lubridate", "randomForest", "shiny", "plotly"))
```

## 2) Run pipeline + models

Dataset requirement:
- Place your source file at `data/stock data.csv`
- Required columns: `date, open, high, low, close, volume, Name`

### Python data + model

```bash
python python/data_pipeline.py
python python/train_model.py
```

### R model

```bash
Rscript r/train_model.R
```

## 3) Launch dashboards

### Python Dash

```bash
python python/dashboard.py
```

### R Shiny

```bash
Rscript r/dashboard.R
```

## 4) Run API

```bash
python python/app_api.py
```

### Endpoints
- `GET /health`
- `POST /predict`

Example `/predict` body:

```json
{
  "features": {
    "return_1d": 0.004,
    "return_5d": 0.011,
    "ma_10": 182.5,
    "ma_20": 178.9,
    "volatility_10": 0.014,
    "volume_change_1d": 0.08
  }
}
```