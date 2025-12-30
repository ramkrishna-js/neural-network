import numpy as np
from utils import load_mnist
from network import NeuralNetwork
import os

def evaluate_model():
    if not os.path.exists("model.pkl"):
        print("Model file not found!")
        return

    x_train, y_train, x_test, y_test = load_mnist()
    
    nn = NeuralNetwork()
    nn.load("model.pkl")
    
    print("Evaluating on test set...")
    predictions = nn.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == y_test)
    
    print(f"Results:")
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Correct Predictions: {np.sum(pred_labels == y_test)}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model()
