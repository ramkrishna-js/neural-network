import numpy as np
from utils import load_mnist, one_hot
from layers import Dense
from activations import ReLU, Softmax
from network import NeuralNetwork

def train_model():
    print("Loading data...")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Preprocess labels
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)
    
    # Define network
    nn = NeuralNetwork()
    nn.add(Dense(28 * 28, 128))
    nn.add(ReLU())
    nn.add(Dense(128, 64))
    nn.add(ReLU())
    nn.add(Dense(64, 10))
    nn.add(Softmax())
    
    # Train
    print("Starting training...")
    nn.train(x_train, y_train_oh, epochs=5, learning_rate=0.1, batch_size=64)
    
    # Save model
    nn.save("model.pkl")
    
    # Evaluate
    print("\nEvaluating...")
    test_output = nn.predict(x_test)
    predictions = np.argmax(test_output, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
