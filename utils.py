import numpy as np
import requests
import gzip
import os

def download_mnist():
    """Downloads MNIST files if they don't exist."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    os.makedirs("data", exist_ok=True)
    
    for file in files:
        path = os.path.join("data", file)
        if not os.path.exists(path):
            print(f"Downloading {file}...")
            r = requests.get(base_url + file, stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

def load_mnist():
    """Loads MNIST data from local files."""
    download_mnist()
    
    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip magic number, num_images, rows, cols
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28) / 255.0

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip magic number, num_items
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = read_images("data/train-images-idx3-ubyte.gz")
    y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    x_test = read_images("data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")
    
    return x_train, y_train, x_test, y_test

def one_hot(y, num_classes=10):
    """Converts labels to one-hot encoding."""
    return np.eye(num_classes)[y]
