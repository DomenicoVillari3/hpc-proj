import numpy as np
import json
import os
import time
import csv
from dotenv import load_dotenv

load_dotenv()



DATA_DIR   = os.getenv("DATASET_DIR")
DATASET    = os.getenv("DATASET_NAME")
RESULTS    = os.path.join(os.getenv("RESULTS_DIR"), "sequential.csv")
os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

  

SEED=int(os.getenv("SEED", 42))
# Iperparametri
EPOCHS     = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
LR0        = float(os.getenv("LEARNING_RATE", 0.01))



# --- Caricamento dataset ---
meta_path = os.path.join(DATA_DIR, f"{DATASET}_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

N_train = meta["N_train"]
N_test  = meta["N_test"]
D       = meta["D"]

print(f"Dataset: {DATASET} N_train={N_train}  N_test={N_test} D={D}")

X_train = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"), dtype=np.float32).reshape(N_train, D)
y_train = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"), dtype=np.float32)
X_test  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"),  dtype=np.float32).reshape(N_test, D)
y_test  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"),  dtype=np.float32)

# --- Funzioni SGD ---
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def compute_loss(X, y, w):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-7, 1 - 1e-7) # per evitare log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_accuracy(X, y, w):
    preds = (sigmoid(X @ w) >= 0.5).astype(np.float32)
    return np.mean(preds == y)

# --- Inizializzazione ---
rng = np.random.default_rng(SEED)
w   = rng.normal(0, 0.01, D).astype(np.float64) # pesi

# --- Training ---
results = []
t_global = 0  # step globale per lr decrescente

print(f"[INFO] Inizio training | epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")
t_start = time.time()

for epoch in range(EPOCHS):
    t_epoch = time.time()

    # Learning rate decrescente per epoca
    lr = LR0 / (1.0 + epoch * 0.1)

    # Shuffle
    idx = rng.permutation(N_train)
    X_shuf = X_train[idx]
    y_shuf = y_train[idx]

    # Mini-batch SGD
    for start in range(0, N_train, BATCH_SIZE):
        X_batch = X_shuf[start:start + BATCH_SIZE]
        y_batch = y_shuf[start:start + BATCH_SIZE]

        
        grad = (X_batch.T @ (sigmoid(X_batch @ w) - y_batch)) / len(y_batch)
        w -= lr * grad
        

    epoch_time = time.time() - t_epoch

    # Valutazione ogni 10 epoche
    if (epoch + 1) % 10 == 0 or epoch == 0:
        loss = compute_loss(X_train, y_train, w)
        acc  = compute_accuracy(X_test, y_test, w)
        print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | acc={acc:.4f} | time={epoch_time:.2f}s")
        results.append({
            "epoch": epoch + 1,
            "loss":  round(float(loss), 6),
            "accuracy": round(float(acc), 6),
            "epoch_time_s": round(epoch_time, 4)
        })

total_time = time.time() - t_start
print(f"\n[DONE] Tempo totale: {total_time:.2f}s")

# --- Salva risultati ---
with open(RESULTS, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy", "epoch_time_s"])
    writer.writeheader()
    writer.writerows(results)

print(f"[DONE] Risultati salvati in {RESULTS}")