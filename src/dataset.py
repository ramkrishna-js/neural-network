import numpy as np
import requests
import gzip
import os
from rich import print as rprint

class Dataset:
    """Handles MNIST and Fashion-MNIST datasets."""
    
    URLS = {
        "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    }

    def __init__(self, name="mnist", data_dir="data"):
        self.name = name.lower()
        self.data_dir = os.path.join(data_dir, self.name)
        self.base_url = self.URLS.get(self.name, self.URLS["mnist"])
        
        self.files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]

    def download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for file in self.files:
            path = os.path.join(self.data_dir, file)
            if not os.path.exists(path):
                rprint(f"[bold yellow]Downloading {self.name} {file}...[/bold yellow]")
                rprint(f"From: {self.base_url + file}")
                try:
                    r = requests.get(self.base_url + file, stream=True)
                    r.raise_for_status()
                    with open(path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk: f.write(chunk)
                except Exception as e:
                    rprint(f"[bold red]Failed to download {file}: {e}[/bold red]")
                    # Handle mirror if needed or just fail
                    raise

    def load(self):
        self.download()
        
        def read_images(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28 * 28) / 255.0

        def read_labels(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

        x_train = read_images(os.path.join(self.data_dir, self.files[0]))
        y_train = read_labels(os.path.join(self.data_dir, self.files[1]))
        x_test = read_images(os.path.join(self.data_dir, self.files[2]))
        y_test = read_labels(os.path.join(self.data_dir, self.files[3]))
        
        return x_train, y_train, x_test, y_test

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

class DataLoader:
    """Iterates over batches of data."""
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = x.shape[0]

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size