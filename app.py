from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load models
with open('models/failure_reason_model.pkl', 'rb') as f:
    reason_model = pickle.load(f)

with open('models/failure_cause_model.pkl', 'rb') as f:
    cause_model = pickle.load(f)

# Load label encoders
with open('models/failure_reason_label_encoder.pkl', 'rb') as f:
    reason_encoder = pickle.load(f)

with open('models/failure_cause_label_encoder.pkl', 'rb') as f:
    cause_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features' in request"}), 400

    # Get predictions (encoded labels)
    reason_encoded = reason_model.predict([features])[0]
    cause_encoded = cause_model.predict([features])[0]

    # Decode to string labels
    reason = reason_encoder.inverse_transform([reason_encoded])[0]
    cause = cause_encoder.inverse_transform([cause_encoded])[0]

    return jsonify({
        "failure_reason": reason,
        "failure_cause": cause
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
