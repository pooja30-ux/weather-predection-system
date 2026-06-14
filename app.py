from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("temperature_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "Weather Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input into DataFrame
    input_df = pd.DataFrame([data])

    # Scale input
    scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled)[0]

    return jsonify({
        "predicted_temperature": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)

    joblib.dump(model, 'temperature_model.pkl')