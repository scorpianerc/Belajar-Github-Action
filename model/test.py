import pandas as pd
import numpy as np
import pickle

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            weights = pickle.load(f)
        print("Success")
        return weights
    except:
        print("Failed")
        return None

def predict(weights, data):
    predictions = np.array([])
    for i in range(data.shape[0]):
        y_pred = np.sign(np.dot(weights, data.to_numpy()[i]))
        predictions = np.append(predictions, y_pred)
    return predictions

def main():
    print("Testing...")
    
    # Load model
    weights = load_model()
    if weights is None:
        return False
    
    # Load data untuk test
    data = pd.read_csv('iris_dataset.csv')
    data.drop(data[data['target'] == 'Iris-virginica'].index, inplace=True)
    data['target'] = data['target'].map({'Iris-setosa': -1, 'Iris-versicolor': 1})
    
    # Sample untuk test
    test_data = data.sample(n=5, random_state=42)
    test_labels = test_data.pop('target')
    
    # Prediksi
    predictions = predict(weights, test_data)
    
    print("Hasil prediksi:")
    for i in range(len(predictions)):
        actual = "Iris-setosa" if test_labels.iloc[i] == -1 else "Iris-versicolor"
        predicted = "Iris-setosa" if predictions[i] == -1 else "Iris-versicolor"
        status = "BENAR" if test_labels.iloc[i] == predictions[i] else "SALAH"
        print(f"{i+1}. Actual: {actual}, Predicted: {predicted} [{status}]")
    
    # Simpan log prediksi
    with open('log.txt', 'w', encoding='utf-8') as flog:
        flog.write("Hasil prediksi lengkap ...")
    
    # Simpan hanya nilai akurasi
    accuracy = sum(test_labels == predictions) / len(test_labels)
    with open('accuracy.txt', 'w') as facc:
        facc.write(str(accuracy))
    
    print("Testing selesai")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
