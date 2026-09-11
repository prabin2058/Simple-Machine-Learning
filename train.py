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

# 1. Load data
df = pd.read_csv(Path(__file__).parent / "data" / "home_prices.csv")
print("Data shape:", df.shape)
print(df.head(), "\n")

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
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

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