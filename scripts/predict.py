import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Dataset
from src.network import NeuralNetwork
from rich.panel import Panel
from rich.columns import Columns
from rich import print as rprint

def img_to_ascii(img_data):
    img = img_data.reshape(28, 28)
    chars = [" ", "░", "▒", "▓", "█"]
    output = ""
    for y in range(0, 28, 2):
        for x in range(0, 28, 1):
            val = img[y, x]
            char = chars[int(val * (len(chars) - 1))]
            output += f"[bold green]{char}[/bold green]" if val > 0.1 else f"[dim white]{char}[/dim white]"
        output += "\n"
    return output

def predict(dataset_name="mnist", mode="both"):
    model_path = f"models/{dataset_name}_model.pkl"
    if not os.path.exists(model_path):
        rprint(f"[bold red]Model {model_path} not found![/bold red]")
        return

    dataset = Dataset(dataset_name)
    _, _, x_test, y_test = dataset.load()
    
    nn = NeuralNetwork()
    nn.load(model_path)
    
    indices = np.random.choice(len(x_test), 5)
    cli_panels = []
    
    if mode in ["png", "both"]:
        plt.figure(figsize=(15, 3))

    for i, idx in enumerate(indices):
        img = x_test[idx]
        label = y_test[idx]
        prediction = nn.predict(img.reshape(1, -1))
        pred_label = np.argmax(prediction)
        
        # CLI
        ascii_art = img_to_ascii(img)
        cli_panels.append(Panel(ascii_art, title=f"P: {pred_label}", subtitle=f"T: {label}"))

        # PNG
        if mode in ["png", "both"]:
            plt.subplot(1, 5, i+1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            plt.title(f"P: {pred_label} (T: {label})")
            plt.axis('off')
    
    if mode in ["cli", "both"]:
        rprint(Columns(cli_panels))
    
    if mode in ["png", "both"]:
        plt.savefig(f"prediction_{dataset_name}.png")
        rprint(f"[bold green]Saved to prediction_{dataset_name}.png[/bold green]")

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    predict(dataset, mode)