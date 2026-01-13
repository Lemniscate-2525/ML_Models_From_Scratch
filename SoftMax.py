import numpy as np

class SoftmaxClassifier:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1

        for _ in range(self.epochs):
            scores = X @ self.W + self.b
            probs = self.softmax(scores)

            dW = X.T @ (probs - y_onehot) / n_samples
            db = np.sum(probs - y_onehot, axis=0, keepdims=True) / n_samples

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        scores = X @ self.W + self.b
        return np.argmax(scores, axis=1)
