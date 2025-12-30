# Neural Network from Scratch 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![MNIST Accuracy](https://img.shields.io/badge/MNIST-97.04%25-green.svg)](#)

A modular implementation of a Neural Network built entirely with Python and NumPy. This project is now organized into a professional package structure and supports multiple datasets.

## 🏗 Architecture
- **Layer-based modularity**: Add as many `Dense` layers as you want.
- **DataLoader**: Efficient mini-batch handling and shuffling.
- **Multiple Datasets**: Support for both standard **MNIST** and **Fashion-MNIST**.

## 📁 Project Structure
```text
.
├── src/            # Core logic (Layers, Network, Dataset)
├── scripts/        # CLI Tools (Train, Predict, Evaluate)
├── models/         # Saved .pkl models
├── data/           # Downloaded datasets
└── README.md
```

## 🚀 Usage

### 1. Training
Train a model on a specific dataset:
```bash
python scripts/train.py mnist
# OR
python scripts/train.py fashion
```

### 2. Prediction
View AI predictions in CLI or PNG:
```bash
python scripts/predict.py fashion cli
```

### 3. Evaluation
Get final accuracy on test set:
```bash
python scripts/evaluate.py fashion
```

## 🌳 Visualization
View the cool project tree:
```bash
python scripts/tree_view.py
```

## 📜 License
MIT
