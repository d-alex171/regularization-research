import numpy as np

class SoftmaxClassifier:
    def __init__(self, learning_rate=0.01, num_classes=3, num_features=2):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = np.zeros((1, num_classes))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # Subtract max to prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        log_likelihood = -np.log(predicted[range(m), actual])
        loss = np.sum(log_likelihood) / m
        return loss

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)

            # Compute loss
            loss = self.cross_entropy_loss(probabilities, y)

            # Backward pass (Gradient Descent)
            m = X.shape[0]
            grad_logits = probabilities
            grad_logits[range(m), y] -= 1  # Gradient of loss with respect to logits
            grad_logits /= m

            # Update weights and bias
            self.weights -= self.learning_rate * np.dot(X.T, grad_logits)
            self.bias -= self.learning_rate * np.sum(grad_logits, axis=0, keepdims=True)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)