from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("nyctaxi.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        passenger_count = float(data['passenger_count'])
        pickup_hour = float(data['pickup_hour'])
        pickup_day = float(data['pickup_day'])
        pickup_month = float(data['pickup_month'])
        distance_km = float(data['distance_km'])
        store_and_fwd_flag_Y = float(data['store_and_fwd_flag_Y'])

        # Prepare feature array for prediction
        features = np.array([[passenger_count, pickup_hour, pickup_day, pickup_month, distance_km, store_and_fwd_flag_Y]])

        # Predict trip duration
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)  # round to 2 decimals

        return jsonify({'duration': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
