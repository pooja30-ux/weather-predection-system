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

# Line Plot (First 50 Samples)
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