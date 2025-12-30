import numpy as np
import pickle
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_gradient(y_true, y_pred):
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

    def train(self, data_loader, epochs, learning_rate):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
        ) as progress:
            
            for epoch in range(epochs):
                epoch_loss = 0
                task_id = progress.add_task(f"Epoch {epoch+1}/{epochs}", total=len(data_loader), loss=0)
                
                for x_batch, y_batch in data_loader:
                    # Forward
                    output = self.predict(x_batch)
                    
                    # Loss
                    batch_loss = cross_entropy_loss(y_batch, output)
                    epoch_loss += batch_loss
                    
                    # Backward
                    gradient = cross_entropy_gradient(y_batch, output)
                    for layer in reversed(self.layers):
                        gradient = layer.backward(gradient, learning_rate)
                    
                    progress.update(task_id, advance=1, loss=batch_loss)
                
                avg_loss = epoch_loss / len(data_loader)
                progress.update(task_id, description=f"Epoch {epoch+1}/{epochs} (Avg Loss: {avg_loss:.4f})")

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.layers = pickle.load(f)
        print(f"Model loaded from {filename}")

import os
