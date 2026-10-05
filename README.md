# Machine Learning & Deep Learning Home Price Prediction

End-to-end Machine Learning project comparing **Linear Regression**, **Decision Tree**, **Random Forest**, and a **Multi-Layer Perceptron (MLP) Neural Network** for housing price prediction, enhanced with **StandardScaler**, **Principal Component Analysis (PCA)**, and complete **training loss & performance graph analysis**.

> 📖 **Full In-Depth Technical Guide**: For complete mathematical explanations, PCA proofs, loss curve derivations, and graph interpretations, see [DOCUMENTATION.md](file:///Users/prabinkarki/Desktop/ML/DOCUMENTATION.md).

---

## 📊 Project Highlights

- **Dataset**: 500 house sales records across 7 predictive features (`area_sqft`, `bedrooms`, `bathrooms`, `age_years`, `garage_spaces`, `location_score`, `distance_to_city_km`).
- **Feature Scaling**: Standard Normalization ($Z$-score scaling) on features and target variables.
- **Dimensionality Reduction (PCA)**: Principal Component Analysis preserving **95% cumulative explained variance**.
- **Models Evaluated**:
  1. **Linear Regression** — Parametric baseline with interpretable feature coefficients.
  2. **Decision Tree Regressor** (`max_depth=5`) — Non-linear tree partitioning.
  3. **Random Forest Regressor** (`n_estimators=200`) — Bagging ensemble of 200 trees.
  4. **Neural Network (MLPRegressor)** (`hidden_layer_sizes=(64, 32)`, ReLU activation, Adam optimizer, iteration loss curve tracking).
- **Evaluation Metrics**: $R^2$ Score (Goodness of fit), MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error).
- **Graph Diagnostics**: 8 detailed visual charts covering distributions, PCA variance, epoch loss descent, model comparison, and actual vs. predicted scatters.

---

## 📈 Graph Analysis Summary

| Graph Name | File / Notebook Source | What It Analyzes |
|---|---|---|
| **PCA Explained Variance** | `notebook.ipynb` | Shows cumulative variance vs. number of components with 95% threshold line. |
| **Neural Network Loss Curve** | `notebook.ipynb` | Tracks MSE loss across epochs/iterations using the Adam optimizer to verify model convergence. |
| **Model $R^2$ Comparison** | `plots/04_r2_comparison.png` & notebook | Bar chart comparing the percentage of variance explained by each model ($R^2$). |
| **Model MAE Comparison** | `notebook.ipynb` | Compares average absolute dollar prediction error across all models. |
| **Model RMSE Comparison** | `notebook.ipynb` | Compares root mean squared errors, highlighting sensitivity to outlier errors. |
| **Actual vs. Predicted** | `plots/05_actual_vs_pred.png` & notebook | Multi-panel scatter plot comparing predictions against ground truth diagonal line $y = x$. |
| **Linear Regression Coefficients** | `plots/03_lr_coefficients.png` & notebook | Horizontal bar chart showing positive (green) and negative (red) feature impacts on price. |
| **Price & Feature Distribution** | `plots/01_price_hist.png`, `02_area_location_scatter.png` | Histogram of sale prices and scatter plots showing correlations for area and location score. |

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Main Training Script
```bash
# Trains models and exports 5 PNG charts into plots/
python train.py

# Optional: Display interactive popup windows
python train.py --show
```

### 3. Run Inference on New Test Houses
```bash
python analyze_new.py
```
Exports predictions and error breakdowns to `predictions_10.csv` and `analysis_10.txt`.

### 4. Interactive Jupyter Notebook
```bash
pip install jupyter
jupyter lab notebook.ipynb
```
Opens interactive workflow with inline PCA scree plots, Neural Network epoch training curves, and interactive model benchmarks.

---

## 📁 Repository Structure

```
ML/
├── data/
│   ├── home_prices.csv               # 500-row primary training dataset
│   └── new_house_data_10_rows.csv    # 10-row new house validation set
├── plots/                            # Auto-generated high-resolution PNG charts
│   ├── 01_price_hist.png             # Target price distribution
│   ├── 02_area_location_scatter.png  # Area & location correlation scatters
│   ├── 03_lr_coefficients.png        # Feature coefficient weights
│   ├── 04_r2_comparison.png          # Model R² comparison bar chart
│   └── 05_actual_vs_pred.png         # Actual vs. predicted scatter plots
├── train.py                          # Main model training & plot generation script
├── analyze_new.py                    # Inference script for new house data
├── notebook.ipynb                    # Jupyter notebook (PCA, Neural Net epochs, plots)
├── predictions_10.csv                # Prediction outputs on 10-house verification set
├── analysis_10.txt                   # Text summary report of 10-house predictions
├── requirements.txt                  # Python dependencies
├── DOCUMENTATION.md                  # Comprehensive technical documentation
└── README.md                         # Project overview and quickstart guide
```
