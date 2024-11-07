from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

with open('modelo_regressao_linear.pickle', 'rb') as file:
    modelo_regressao_linear = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required_fields = ['sqft_living', 'grade', 'bathrooms', 'sqft_above']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Por favor, forneça o valor de '{field}'."}), 400

    try:
        features = np.array([[float(data['sqft_living']),
                              float(data['grade']),
                              float(data['bathrooms']),
                              float(data['sqft_above'])]])
        
        features_scaled = scaler.transform(features)
    except ValueError as e:
        return jsonify({"error": f"Valor inválido: {str(e)}"}), 400

    predicted_price = modelo_regressao_linear.predict(features_scaled)[0]

    return jsonify({"predicted_price": predicted_price})

if __name__ == '__main__':
    app.run(debug=True)

# curl -X POST -H "Content-Type: application/json" -d '{"sqft_living": 2000, "grade": 7, "bathrooms": 2, "sqft_above": 1800}' http://127.0.0.1:5000/predict