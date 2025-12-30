import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Dataset
from src.network import NeuralNetwork
from rich import print as rprint

def evaluate(dataset_name="mnist"):
    model_path = f"models/{dataset_name}_model.pkl"
    if not os.path.exists(model_path):
        rprint(f"[bold red]Model {model_path} not found![/bold red]")
        return

    dataset = Dataset(dataset_name)
    _, _, x_test, y_test = dataset.load()
    
    nn = NeuralNetwork()
    nn.load(model_path)
    
    predictions = nn.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(pred_labels == y_test)
    
    rprint(f"[bold green]Dataset: {dataset_name.upper()}[/bold green]")
    rprint(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    evaluate(dataset)