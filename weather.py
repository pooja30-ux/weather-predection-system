# ============================================
# WEATHER PREDICTION MODEL (IMPROVED VERSION)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("1.csv")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

# Handle missing values
df = df.dropna()

print("First 5 Rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ----------------------------
# 2. FEATURES & TARGET
# ----------------------------
features = [
    "humidity",
    "windspeed",
    "sealevelpressure",
    "cloudcover",
    "solarradiation"
]

target = "temp"

X = df[features]
y = df[target]

# ----------------------------
# 3. TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# 4. MODEL TRAINING
# ----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# 5. PREDICTIONS
# ----------------------------
pred = model.predict(X_test)

print("\nFirst 10 Predictions:", pred[:10])

# ----------------------------
# 6. EVALUATION
# ----------------------------
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nMODEL PERFORMANCE")
print("------------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))

print("\nTrain Score:", round(model.score(X_train, y_train), 3))
print("Test Score :", round(model.score(X_test, y_test), 3))

# ----------------------------
# 7. FEATURE IMPORTANCE
# ----------------------------
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.bar(importance["Feature"], importance["Importance"])
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.grid(True)
plt.show()

# ----------------------------
# 8. ACTUAL VS PREDICTED
# ----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# ----------------------------
# 9. LINE PLOT (FIRST 50)
# ----------------------------
plt.figure(figsize=(10, 5))

plt.plot(y_test.values[:50], label="Actual", marker="o")
plt.plot(pred[:50], label="Predicted", marker="x")

plt.title("Actual vs Predicted (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 10. SAVE MODEL
# ----------------------------
joblib.dump(model, "temperature_model.pkl")
print("\nModel saved as temperature_model.pkl")

# ----------------------------
# 11. LOAD & TEST (OPTIONAL)
# ----------------------------
loaded_model = joblib.load("temperature_model.pkl")

new_data = pd.DataFrame({
    "humidity": [70],
    "windspeed": [10],
    "sealevelpressure": [1012],
    "cloudcover": [40],
    "solarradiation": [600]
})

prediction = loaded_model.predict(new_data)
print("\nPredicted Temperature for new data:", prediction[0])