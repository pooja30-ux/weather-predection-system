import pandas as pd  # alias for pandas used as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv("1.csv")


print("First 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

# Features
X = df[
    [
        "humidity",
        "windspeed",
        "sealevelpressure",
        "cloudcover",
        "solarradiation"
    ]
]


y = df["temp"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)


print("\nFirst 10 Predictions:")
print(pred[:10])


mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Performance")
print("------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:")
print(
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)
 
# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
import joblib

joblib.dump(model, "temperature_model.pkl")
print("Model saved successfully!")
model = joblib.load("temperature_model.pkl")

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(
    y_test.values[:50],
    label="Actual",
    marker="o"
)

plt.plot(
    pred[:50],
    label="Predicted",
    marker="x"
)

plt.title("Actual vs Predicted (First 50 Test Samples)")
plt.xlabel("Sample Number")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()