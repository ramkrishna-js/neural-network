import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Dataset, DataLoader, one_hot
from src.layers import Dense
from src.activations import ReLU, Softmax
from src.network import NeuralNetwork
from rich import print as rprint

def run_training(dataset_name="mnist"):
    rprint(f"[bold cyan]Initializing {dataset_name.upper()} project...[/bold cyan]")
    
    # Load data
    dataset = Dataset(dataset_name)
    x_train, y_train, x_test, y_test = dataset.load()
    
    y_train_oh = one_hot(y_train)
    data_loader = DataLoader(x_train, y_train_oh, batch_size=64)
    
    # Define network
    nn = NeuralNetwork()
    nn.add(Dense(28 * 28, 128))
    nn.add(ReLU())
    nn.add(Dense(128, 64))
    nn.add(ReLU())
    nn.add(Dense(64, 10))
    nn.add(Softmax())
    
    # Train
    rprint("[bold green]Starting training...[/bold green]")
    nn.train(data_loader, epochs=5, learning_rate=0.1)
    
    # Save
    model_path = f"models/{dataset_name}_model.pkl"
    nn.save(model_path)
    
    # Quick eval
    rprint("\n[bold cyan]Evaluating...[/bold cyan]")
    test_output = nn.predict(x_test)
    predictions = np.argmax(test_output, axis=1)
    accuracy = np.mean(predictions == y_test)
    rprint(f"[bold green]Test Accuracy: {accuracy * 100:.2f}%[/bold green]")

if __name__ == "__main__":
    dataset_name = "mnist"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    run_training(dataset_name)