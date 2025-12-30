import numpy as np

class Activation:
    def forward(self, input):
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)

class Softmax(Activation):
    def forward(self, input):
        # Numeric stability: subtract max
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities
    
    def backward(self, output_gradient, learning_rate):
        # Softmax backward is often combined with Cross-Entropy for simplicity
        # But here we provide a generic version if needed.
        # However, in most implementations, we handle Softmax+Loss together.
        return output_gradient 
