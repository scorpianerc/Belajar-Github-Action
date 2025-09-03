import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def hitung_cost_gradient(W, X, Y, regularization):
    jarak = 1 - (Y * np.dot(X, W))
    dw = np.zeros(len(W))
    if max(0, jarak) == 0:
        di = W
    else:
        di = W - (regularization * Y * X)
    dw += di
    return dw

def sgd(data_latih, label_latih, learning_rate=0.000001, max_epoch=1000, regularization=10000):
    print("Training...")
    data_latih = data_latih.to_numpy()
    label_latih = label_latih.to_numpy()
    bobot = np.zeros(data_latih.shape[1])
    
    for epoch in range(1, max_epoch):
        X, Y = shuffle(data_latih, label_latih, random_state=101)
        for index, x in enumerate(X):
            delta = hitung_cost_gradient(bobot, x, Y[index], regularization)
            bobot = bobot - (learning_rate * delta)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{max_epoch}")
    
    print("Training selesai")
    return bobot

def testing(W, data_uji):
    prediksi = np.array([])
    for i in range(data_uji.shape[0]):
        y_prediksi = np.sign(np.dot(W, data_uji.to_numpy()[i]))
        prediksi = np.append(prediksi, y_prediksi)
    return prediksi

def main():
    data = pd.read_csv('iris_dataset.csv')
    
    # Remove Iris-virginica
    data.drop(data[data['target'] == 'Iris-virginica'].index, inplace=True)
    
    # Map target
    data['target'] = data['target'].map({'Iris-setosa': -1, 'Iris-versicolor': 1})
    
    # Split data
    data_latih, data_uji = train_test_split(data, test_size=0.2, random_state=42)
    label_latih = data_latih.pop('target')
    label_uji = data_uji.pop('target')
    
    print(f"Data training: {data_latih.shape[0]} samples")
    print(f"Data testing: {data_uji.shape[0]} samples")
    
    # Training
    bobot = sgd(data_latih, label_latih)
    
    # Testing
    print("Testing...")
    prediksi = testing(bobot, data_uji)
    
    # Evaluasi
    accuracy = accuracy_score(label_uji, prediksi)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Simpan model
    with open('model.pkl', 'wb') as f:
        pickle.dump(bobot, f)
    print("Model disimpan ke model.pkl")
    
    # Simpan accuracy
    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    
    return accuracy

if __name__ == "__main__":
    main()
