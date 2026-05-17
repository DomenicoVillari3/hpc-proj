import numpy as np


# --- Funzioni comuni ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def compute_loss(X, y, w):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_accuracy(X, y, w):
    preds = (sigmoid(X @ w) >= 0.5).astype(np.float32)
    return np.mean(preds == y)