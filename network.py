import numpy as np
import pickle

def cross_entropy_loss(y_true, y_pred):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_gradient(y_true, y_pred):
    # This is the gradient of Loss wrt the input of Softmax
    return (y_pred - y_true) / y_true.shape[0]

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate, batch_size=32):
        n_samples = x_train.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.predict(x_batch)
                
                # Loss
                epoch_loss += cross_entropy_loss(y_batch, output)
                
                # Backward
                gradient = cross_entropy_gradient(y_batch, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (n_samples // batch_size):.4f}")

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)
        print(f"Model loaded from {filename}")