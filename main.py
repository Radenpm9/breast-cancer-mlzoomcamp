from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the package once when the server starts
with open('cancer_model_v1.pkl', 'rb') as f:
    package = pickle.load(f)
    model = package["model"]
    labels = package["labels"]

@app.route('/predict', methods=['POST'])
def predict():
    # Receive data as JSON
    data = request.get_json()
    
    # Convert input to 2D numpy array
    features = np.array(data['features']).reshape(1, -1)
    
    # Generate prediction and probability
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 0] # Prob of Malignant
    
    result = {
        "diagnosis": str(labels[prediction[0]]),
        "malignancy_probability": round(float(probability[0]), 4)
    }
    return jsonify(result)

@app.route('/', methods=['GET'])
def home():
    return {
        "status": "Success",
        "message": "Breast Cancer Diagnostic API is running!",
        "version": "1.0"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)