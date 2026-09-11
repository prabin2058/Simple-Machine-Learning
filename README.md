# ML Home Price Prediction

Simple ML models to predict home prices. Compares Linear Regression, Decision Tree, Random Forest.

## Dataset

`data/home_prices.csv` — 500 rows, 8 columns (7 features + price)

| Feature | Description |
|---|---|
| area_sqft | House area in sq ft |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| age_years | Age of house |
| garage_spaces | Garage capacity |
| location_score | Location rating (0-10) |
| distance_to_city_km | Distance to city center |
| price | Target — home price ($) |

## Models

- **Linear Regression** — baseline
- **Decision Tree** (`max_depth=5`) — interpretable
- **Random Forest** (`n_estimators=200`) — best R2 usually

Evaluated on 80/20 split (`random_state=42`) with MAE, RMSE, R2. Also prints LR coefficients and predicts example house (2000 sqft, 3bd/2ba).

## Requirements

- Python 3.10+ (tested on 3.14.6)
- See `requirements.txt`: `pandas`, `numpy`, `scikit-learn`

## Run

```bash
# 1. create venv (required on Homebrew Python — PEP 668)
python3 -m venv .venv
source .venv/bin/activate

# 2. install deps
pip install -r requirements.txt

# 3. run
python train.py
```

Expected output:
```
Data shape: (500, 8)
Model comparison:
            Model   MAE  RMSE    R2
...
Best model: Random Forest
Linear Regression coefficients ...
Linear Regression predicted price ...  $...
Decision Tree predicted price ...      $...
Random Forest predicted price ...      $...
```

## Project Structure

```
ML/
├── train.py              # train + evaluate 3 models
├── data/
│   └── home_prices.csv   # 500-row dataset
├── requirements.txt
└── README.md
```

## Troubleshooting

**`error: externally-managed-environment`**
→ Homebrew Python blocks system pip (PEP 668). Use venv as above. Don't use `--break-system-packages`.

**`FileNotFoundError: home_prices.csv`**
→ Fixed in `train.py:16` to `data/home_prices.csv`. If you moved CSV, update path there.

**`ModuleNotFoundError: pandas/sklearn`**
→ `source .venv/bin/activate` then `pip install -r requirements.txt`
