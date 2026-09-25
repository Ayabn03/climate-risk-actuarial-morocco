import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv("base_finale_modelisation.csv")
X_cols = ["SPI_12", "TXx_anomalie", "R95p_pct", "FFDI", "humidite_pct"]
X = df[X_cols].values
y = df["ratio_SP"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "RLM": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
  model.fit(X_scaled, y)
  y_pred = model.predict(X_scaled)
  rmse = np.sqrt(mean_squared_error(y, y_pred))
  mae = mean_absolute_error(y, y_pred)
  r2 = r2_score(y, y_pred)
  print(f"{name}: RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

rf = models["Random Forest"]
importances = (
    pd.DataFrame({"Variable": X_cols, "Importance": rf.feature_importances_})
    .sort_values("Importance", ascending=False)
)
print(importances)
