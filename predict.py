import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist
from network import NeuralNetwork
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich import print as rprint

def img_to_ascii(img_data):
    """Converts a 28x28 image to a colored CLI string."""
    img = img_data.reshape(28, 28)
    chars = [" ", "░", "▒", "▓", "█"]
    output = ""
    # Downsample to 14x14 for better terminal fit
    for y in range(0, 28, 2):
        for x in range(0, 28, 1):
            val = img[y, x]
            char_idx = int(val * (len(chars) - 1))
            char = chars[char_idx]
            # Use green color for the "cool" look
            if val > 0.1:
                output += f"[bold green]{char}[/bold green]"
            else:
                output += f"[dim black]{char}[/dim black]"
        output += "\n"
    return output

def predict_random(mode="both"):
    if not os.path.exists("model.pkl"):
        rprint("[bold red]Model file not found! Please run train.py first.[/bold red]")
        return

    rprint("[bold cyan]Loading data and model...[/bold cyan]")
    _, _, x_test, y_test = load_mnist()
    
    nn = NeuralNetwork()
    nn.load("model.pkl")
    
    # Pick random images
    indices = np.random.choice(len(x_test), 5)
    
    cli_panels = []
    
    if mode in ["png", "both"]:
        plt.figure(figsize=(12, 3))

    for i, idx in enumerate(indices):
        img = x_test[idx]
        label = y_test[idx]
        
        prediction = nn.predict(img.reshape(1, -1))
        pred_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # CLI Mode
        ascii_art = img_to_ascii(img)
        status_color = "green" if pred_label == label else "red"
        panel = Panel(
            ascii_art,
            title=f"[bold {status_color}]Pred: {pred_label}[/bold {status_color}]",
            subtitle=f"True: {label} ({confidence:.1f}%)",
            expand=False
        )
        cli_panels.append(panel)

        # PNG Mode
        if mode in ["png", "both"]:
            plt.subplot(1, 5, i+1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            plt.title(f"P: {pred_label} (T: {label})")
            plt.axis('off')
    
    # Display CLI
    if mode in ["cli", "both"]:
        rprint("\n[bold green]--- CLI Prediction View ---[/bold green]")
        rprint(Columns(cli_panels))
    
    # Save PNG
    if mode in ["png", "both"]:
        rprint("\n[bold cyan]Saving prediction to 'prediction.png'...[/bold cyan]")
        plt.savefig("prediction.png")
        rprint("[bold green]Done! Open 'prediction.png' to see high-res results.[/bold green]")

if __name__ == "__main__":
    # Check for CLI arguments
    mode = "both"
    if len(sys.argv) > 1:
        if sys.argv[1] in ["cli", "png", "both"]:
            mode = sys.argv[1]
    
    predict_random(mode)
