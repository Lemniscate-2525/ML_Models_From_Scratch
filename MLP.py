import numpy as np

class MLP:
    def __init__(self, layer_sizes, lr=0.01):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.params = self._init_params()

    def _init_params(self):
        params = {}
        for i in range(len(self.layer_sizes) - 1):
            params[f"W{i}"] = np.random.randn(
                self.layer_sizes[i], self.layer_sizes[i+1]
            ) * np.sqrt(2 / self.layer_sizes[i])
            params[f"b{i}"] = np.zeros((1, self.layer_sizes[i+1]))
        return params

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        cache = {"A0": X}
        L = len(self.layer_sizes) - 1

        for i in range(L - 1):
            Z = cache[f"A{i}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self.relu(Z)
            cache[f"Z{i+1}"], cache[f"A{i+1}"] = Z, A

        ZL = cache[f"A{L-1}"] @ self.params[f"W{L-1}"] + self.params[f"b{L-1}"]
        AL = self.softmax(ZL)
        cache[f"Z{L}"], cache[f"A{L}"] = ZL, AL

        return AL, cache

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(m), y_true])) / m

    def backward(self, cache, y_true):
        grads = {}
        m = y_true.shape[0]
        L = len(self.layer_sizes) - 1

        y_onehot = np.zeros_like(cache[f"A{L}"])
        y_onehot[np.arange(m), y_true] = 1

        dZ = cache[f"A{L}"] - y_onehot

        for i in reversed(range(L)):
            grads[f"W{i}"] = cache[f"A{i}"].T @ dZ / m
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA = dZ @ self.params[f"W{i}"].T
                dZ = dA * self.relu_derivative(cache[f"Z{i}"])

        return grads

    def update(self, grads):
        for k in self.params:
            self.params[k] -= self.lr * grads[k]

    def train(self, X, y, epochs=100):
        for e in range(epochs):
            y_pred, cache = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            grads = self.backward(cache, y)
            self.update(grads)

            if e % 10 == 0:
                print(f"Epoch {e}, Loss: {loss:.4f}")
