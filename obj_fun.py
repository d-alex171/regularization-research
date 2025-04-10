import numpy as np

class SoftmaxClassifier:
    def __init__(self, learning_rate=0.1, num_classes=3, num_features=2):
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = np.zeros((1, num_classes))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # Subtract max to prevent overflow
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)

    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        epsilon = 1e-12
        log_likelihood = -np.log(np.clip(predicted[range(m), actual], epsilon, 1.0))
        loss = np.sum(log_likelihood) / m
        return loss
    
    def update_weights(self, X, grad_logits):
        self.weights -= self.learning_rate * np.dot(X.T, grad_logits) 
    

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
            self.update_weights(X,grad_logits)
            self.bias -= self.learning_rate * np.sum(grad_logits, axis=0, keepdims=True)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)


class SoftmaxClassifierL2(SoftmaxClassifier): 
    def __init__(self, learning_rate=0.1, num_classes=3, num_features=2, lammy=0.3):
        super().__init__(learning_rate, num_classes, num_features)
        self.lammy = lammy

    
    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        epsilon = 1e-12
        log_likelihood = -np.log(np.clip(predicted[range(m), actual], epsilon, 1.0))
        loss = np.sum(log_likelihood) / m + self.lammy * np.sum(self.weights ** 2)
        return loss
    
    def update_weights(self, X, grad_logits):
        self.weights -= self.learning_rate * np.dot(X.T, grad_logits) + 2 * self.lammy * self.weights


class SoftmaxClassifierL1(SoftmaxClassifier): 
    def __init__(self, learning_rate=0.1, num_classes=3, num_features=2, lammy=0.3):
        super().__init__(learning_rate, num_classes, num_features)
        self.lammy = lammy

    
    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        epsilon = 1e-12
        log_likelihood = -np.log(np.clip(predicted[range(m), actual], epsilon, 1.0))
        loss = np.sum(log_likelihood) / m + self.lammy * np.sum(np.abs(self.weights))
        return loss
    
    def update_weights(self, X, grad_logits):
        self.weights -= self.learning_rate * np.dot(X.T, grad_logits) + 2 * self.lammy * self.weights
    


class SoftmaxClassifierL0(SoftmaxClassifier): 
    def __init__(self, learning_rate=0.1, num_classes=3, num_features=2, lammy=0.3, epsilon=1e-5):
        super().__init__(learning_rate, num_classes, num_features)
        self.lammy = lammy
        self.epsilon = epsilon

    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        epsilon = 1e-12
        log_likelihood = -np.log(np.clip(predicted[range(m), actual], epsilon, 1.0))
        l1_reg = self.lammy * np.sum(np.abs(self.weights))
        loss = np.sum(log_likelihood) / m + l1_reg
        return loss
    
    def update_weights(self, X, grad_logits):
        self.weights -= self.learning_rate * (np.dot(X.T, grad_logits) + self.lammy * np.sign(self.weights))


class SoftmaxClassifierElasticNet(SoftmaxClassifier):
    def __init__(self, learning_rate=0.1, num_classes=3, num_features=2, lammy=0.3, epsilon=1e-5):
        super().__init__(learning_rate, num_classes, num_features)
        self.lammy = lammy
        self.epsilon = epsilon

    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]  # Number of samples
        epsilon = 1e-12
        log_likelihood = -np.log(np.clip(predicted[range(m), actual], epsilon, 1.0))
        l1_reg = self.lammy * np.sum(np.abs(self.weights))
        l2_reg = self.lammy * np.sum(self.weights ** 2)
        loss = np.sum(log_likelihood) / m + l1_reg + l2_reg
        return loss

    def update_weights(self, X, grad_logits):
        self.weights -= self.learning_rate * (np.dot(X.T, grad_logits) + self.lammy * np.sign(self.weights)) + 2 * self.lammy * self.weights


    
    
    
