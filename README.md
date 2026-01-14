# Neural Network from Scratch 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![MNIST Accuracy](https://img.shields.io/badge/MNIST-97.04%25-green.svg)](#)
[![Fashion-MNIST](https://img.shields.io/badge/Fashion--MNIST-85.91%25-orange.svg)](#)

A modular implementation of a Neural Network built entirely with Python and NumPy. This project features a scratch-built engine supporting both Computer Vision (image recognition) and Natural Language Processing (chatting with notes).

## 🌟 Key Features
- **Computer Vision**: Recognize handwritten digits (MNIST) and clothing items (Fashion-MNIST).
- **Study Buddy AI**: Upload your study notes (PDF, PPTX, TXT) and chat with a model trained specifically on them.
- **Pure NumPy**: No PyTorch or TensorFlow. Manual implementation of backpropagation through time (BPTT).
- **Advanced Optimizers**: Includes Mini-batch Gradient Descent and Adagrad.
- **Professional CLI**: Beautiful terminal UI with progress bars, trees, and ASCII visualizations.

## 🏗 Architecture

### Computer Vision (MLP)
- **Modular Layers**: Flexible `Dense` layers with He Initialization.
- **Activations**: ReLU for hidden layers, Softmax for classification.

### NLP Engine (RNN)
- **Recurrent Neural Network**: Character-level sequence modeling with `tanh` memory states.
- **File Processors**: Automatic text extraction from `.pdf`, `.pptx`, and `.txt`.

## 🚀 Usage

### 📊 Computer Vision (MNIST / Fashion)
1. **Train**: `python scripts/train.py [mnist|fashion]`
2. **Predict**: `python scripts/predict.py [mnist|fashion] [cli|png|both]`
3. **Evaluate**: `python scripts/evaluate.py [mnist|fashion]`

### 📚 Study Buddy AI (NLP)
1. **Upload**: Place your notes in the `my_notes/` folder.
2. **Train**:
   ```bash
   python scripts/study_buddy.py train my_notes/your_notes.pdf
   ```
3. **Chat**:
   ```bash
   python scripts/study_buddy.py  chat
   ```

## 🌳 Visualization
View the professional project structure:
```bash
python scripts/tree_view.py
```

## 📁 Project Structure
```text
.
├── src/            # Core engine (Layers, RNN, Processors)
├── scripts/        # CLI Tools (Train, Predict, Study Buddy)
├── models/         # Trained weight storage
├── my_notes/       # Your personal note repository
└── README.md
```

## 📦 Installation
```bash
git clone https://github.com/ramkrishna-js/neural-network.git
cd "neural network"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📜 License
MIT
