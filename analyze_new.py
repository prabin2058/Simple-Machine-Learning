"""
Analyze new 10-row houses with same 3 models trained on home_prices.csv
No graphs — table + metrics + simple language.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).parent
train_path = BASE / "data" / "home_prices.csv"
new_path = BASE / "data" / "new_house_data_10_rows.csv"

# 1. Train same models on 500
train_df = pd.read_csv(train_path)
X_train_full = train_df.drop(columns=["price"])
y_train_full = train_df["price"]

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}
for m in models.values():
    m.fit(X_train_full, y_train_full)

# 2. Load new 10
new_df = pd.read_csv(new_path)
print(f"New data shape: {new_df.shape}")
print(new_df.head(), "\n")

has_price = "price" in new_df.columns
X_new = new_df.drop(columns=["price"]) if has_price else new_df
y_true = new_df["price"] if has_price else None

# 3. Predict
results = {}
for name, model in models.items():
    results[name] = model.predict(X_new)

# 4. Build output table
out = new_df.copy()
for name, preds in results.items():
    out[f"pred_{name.split()[0].lower()}_linear" if "Linear" in name else f"pred_{name.split()[0].lower()}_{'tree' if 'Tree' in name else 'forest'}"] = np.round(preds).astype(int)

# normalize names for ease: pred_lr, pred_dt, pred_rf
out.rename(columns={
    "pred_linear_linear": "pred_lr",
    "pred_decision_tree": "pred_dt",
    "pred_random_forest": "pred_rf"
}, inplace=True)

for col in ["pred_lr", "pred_dt", "pred_rf"]:
    if has_price:
        out[f"err_{col.split('_')[1]}"] = (out[col] - y_true).abs()
        out[f"pct_{col.split('_')[1]}"] = (out[f"err_{col.split('_')[1]}"] / y_true * 100).round(1)

# 5. Print per-row table
cols_show = ["area_sqft", "bedrooms", "location_score", "price"] if has_price else list(X_new.columns)
cols_show += ["pred_lr", "pred_dt", "pred_rf"]
if has_price:
    cols_show += ["err_lr", "err_dt", "err_rf"]

print("Per-house predictions (10 rows):\n")
print(out[cols_show].to_string(index=False))
print()

# 6. Aggregate metrics on 10 (if true price exists)
if has_price:
    print("Aggregate on new 10 (lower MAE/RMSE better, higher R2 better):\n")
    agg = []
    for name, preds in results.items():
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        agg.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    agg_df = pd.DataFrame(agg).sort_values("R2", ascending=False)
    print(agg_df.to_string(index=False))
    print(f"\nBest on new 10: {agg_df.iloc[0]['Model']}")
    # also compare to original test (0.939/0.891/0.783)
    print("\nvs original 100-test: Linear 0.939/20267, RF 0.891/28045, DT 0.783/37709")

    # worst houses
    print("\nWorst predicted houses (largest Linear error):")
    worst = out.sort_values("err_lr", ascending=False).head(3)
    print(worst[["area_sqft", "location_score", "distance_to_city_km", "price", "pred_lr", "err_lr", "pct_lr"]].to_string(index=False))

# 7. Save
out.to_csv(BASE / "predictions_10.csv", index=False)
print(f"\nSaved predictions_10.csv ({len(out)} rows)")

# 8. Simple language summary
summary = f"""
Simple analysis of new 10 houses
--------------------------------
You gave 10 new houses with true price. We used same 3 models learned from 500 houses.

How result came:
- Model learned rule: price ~ 153*area + 12495*location + 8149*bedrooms ... (from train.py)
- For each new house, plug its 7 numbers into rule -> predicted price
- Compare predicted vs true price -> error

Example: first new house 2385 sqft, location 7.42, true $428650
  -> Linear predicted ${int(out.iloc[0]['pred_lr']):,} error ${int(out.iloc[0]['err_lr']):,}
  -> Random Forest ${int(out.iloc[0]['pred_rf']):,} error ${int(out.iloc[0]['err_rf']):,}
  -> Decision Tree ${int(out.iloc[0]['pred_dt']):,} error ${int(out.iloc[0]['err_dt']):,}

Best on this 10: {agg_df.iloc[0]['Model'] if has_price else 'N/A'} (highest R2, lowest avg error)
If Linear still best, it means simple straight-line rule still works for these new houses.
If error much bigger than 20k avg seen before, it means these 10 are unusual (maybe very new or far location).

Files: predictions_10.csv has all preds + errors per house.
"""
print(summary)
with open(BASE / "analysis_10.txt", "w") as f:
    f.write(f"Per-house table:\n{out[cols_show].to_string(index=False)}\n\n")
    if has_price:
        f.write(f"Aggregate on 10:\n{agg_df.to_string(index=False)}\n\n")
    f.write(summary)
print("Saved analysis_10.txt")
