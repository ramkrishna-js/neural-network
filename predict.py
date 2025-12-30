import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist
from network import NeuralNetwork
import os

def predict_random():
    if not os.path.exists("model.pkl"):
        print("Model file not found! Please run train.py first.")
        return

    print("Loading data and model...")
    _, _, x_test, y_test = load_mnist()
    
    nn = NeuralNetwork()
    nn.load("model.pkl")
    
    # Pick 5 random images
    indices = np.random.choice(len(x_test), 5)
    
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        img = x_test[idx]
        label = y_test[idx]
        
        prediction = nn.predict(img.reshape(1, -1))
        pred_label = np.argmax(prediction)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.title(f"P: {pred_label} (T: {label})")
        plt.axis('off')
    
        
    
        print("Saving prediction to 'prediction.png'...")
    
        plt.savefig("prediction.png")
    
        print("Done! Open 'prediction.png' to see the results.")
    
    
    
    if __name__ == "__main__":
    
        predict_random()
    
    