from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model saat API dijalankan
with open('model.pkl', 'rb') as f:
    weights = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Input: {"data": [sepal_length, sepal_width, petal_length, petal_width]}
    content = request.json
    features = content.get('data')
    if not features or len(features) != 4:
        return jsonify({'error': 'Input harus berupa list 4 angka.'}), 400

    # Prediksi
    y_pred = int(np.sign(np.dot(weights, np.array(features))))
    label = 'Iris-setosa' if y_pred == -1 else 'Iris-versicolor'
    return jsonify({'prediction': y_pred, 'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)