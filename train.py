"""
Simple Machine Learning models to predict home prices.
Dataset: home_prices.csv
Models: Linear Regression, Decision Tree, Random Forest
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse

parser = argparse.ArgumentParser(description="Train home price models and generate plots")
parser.add_argument("--show", action="store_true", help="also display plots in window (needs GUI)")
args = parser.parse_args()

import matplotlib
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

def save_and_maybe_show(fname):
    plt.savefig(PLOTS / fname, dpi=150)
    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f"[warn] --show failed: {e}")
    plt.close()

# 1. Load data
df = pd.read_csv(BASE / "data" / "home_prices.csv")
print("Data shape:", df.shape)
print(df.head(), "\n")

# --- Graph 1: Price distribution ---
plt.figure(figsize=(7, 4))
plt.hist(df["price"], bins=30, edgecolor="black")
plt.title("Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.tight_layout()
save_and_maybe_show("01_price_hist.png")

# --- Graph 2: area vs price + location vs price ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df["area_sqft"], df["price"], alpha=0.5, s=12)
axes[0].set_title("Area vs Price")
axes[0].set_xlabel("area_sqft")
axes[0].set_ylabel("price")
axes[1].scatter(df["location_score"], df["price"], alpha=0.5, s=12, color="green")
axes[1].set_title("Location Score vs Price")
axes[1].set_xlabel("location_score")
plt.tight_layout()
save_and_maybe_show("02_area_location_scatter.png")

# 2. Split features / target
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

# 4. Train, predict, evaluate
results = []
preds_dict = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_dict[name] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("Model comparison:\n")
print(results_df.to_string(index=False))

# 5. Show feature importance / coefficients for the best-performing simple model
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")

lr = models["Linear Regression"]
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nLinear Regression coefficients (impact on price):")
print(coef_df.to_string(index=False))

# --- Graph 3: LR coefficients ---
plt.figure(figsize=(7, 4))
colors = ["green" if c > 0 else "red" for c in coef_df["Coefficient"]]
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
plt.title("Linear Regression Coefficients")
plt.xlabel("Coefficient")
plt.tight_layout()
save_and_maybe_show("03_lr_coefficients.png")

# --- Graph 4: Model R2 comparison ---
plt.figure(figsize=(6, 4))
plt.bar(results_df["Model"], results_df["R2"], color=["#4CAF50", "#2196F3", "#FF9800"])
plt.title("Model R2 Comparison (higher better)")
plt.ylabel("R2")
plt.ylim(0, 1)
for i, v in enumerate(results_df["R2"]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
save_and_maybe_show("04_r2_comparison.png")

# --- Graph 5: Actual vs Predicted ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
for ax, (name, preds) in zip(axes, preds_dict.items()):
    ax.scatter(y_test, preds, alpha=0.5, s=12)
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_title(f"{name}\nR2={r2_score(y_test, preds):.3f}")
    ax.set_xlabel("Actual price")
axes[0].set_ylabel("Predicted price")
plt.tight_layout()
save_and_maybe_show("05_actual_vs_pred.png")

print(f"\nPlots saved to {PLOTS}/ (5 PNGs)")

# 6. Example: predict price for a new house
new_house = pd.DataFrame([{
    "area_sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "garage_spaces": 2,
    "location_score": 7.5,
    "distance_to_city_km": 8.0
}])

for name, model in models.items():
    pred_price = model.predict(new_house)[0]
    print(f"\n{name} predicted price for example house: ${pred_price:,.0f}")
