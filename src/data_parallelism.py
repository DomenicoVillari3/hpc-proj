import numpy as np
import json
import os
import time
import csv
from mpi4py import MPI
from dotenv import load_dotenv

from utils import sigmoid, compute_loss, compute_accuracy

load_dotenv()

# --- MPI Init ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

DATA_DIR   = os.getenv("DATASET_DIR")
DATASET    = os.getenv("DATASET_NAME")
RESULTS = os.path.join(os.getenv("RESULTS_DIR"), f"data_parallelism/data_parallel_{DATASET}_np{size}.csv")

SEED       = int(os.getenv("SEED", 42))
EPOCHS     = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
LR0        = float(os.getenv("LEARNING_RATE", 0.1))



# --- Caricamento metadata (tutti i rank) ---
meta_path = os.path.join(DATA_DIR, f"{DATASET}_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

N_train = meta["N_train"]
N_test  = meta["N_test"]
D       = meta["D"]

# --- Partizionamento dataset per rank ---
# Ogni rank carica solo il proprio shard
shard_size = N_train // size
start_idx  = rank * shard_size
end_idx    = start_idx + shard_size if rank < size - 1 else N_train

X_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"), dtype=np.float32).reshape(N_train, D)
y_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"), dtype=np.float32)

X_shard = X_full[start_idx:end_idx].copy()
y_shard = y_full[start_idx:end_idx].copy()
N_shard = len(y_shard)

# Test set caricato solo dal rank 0
if rank == 0:
    X_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"),  dtype=np.float32).reshape(N_test, D)
    y_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"),  dtype=np.float32)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

if rank == 0:
    print(f"[INFO] Dataset: {DATASET} | N_train={N_train} | N_test={N_test} | D={D}")
    print(f"[INFO] Ranks: {size} | shard per rank: ~{shard_size} | epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")

# --- Inizializzazione pesi (identica su tutti i rank) ---
rng = np.random.default_rng(SEED)
w   = rng.normal(0, 0.01, D).astype(np.float64)

# --- Training ---
results    = []
grad_buf   = np.zeros(D, dtype=np.float64)  # buffer Allreduce

comm.Barrier()
t_start = time.time()

for epoch in range(EPOCHS):
    t_epoch = time.time()
    lr = LR0 / (1.0 + epoch * 0.1)

    # Shuffle locale (seed diverso per rank)
    rng_local = np.random.default_rng(SEED + epoch * 100 + rank)
    idx    = rng_local.permutation(N_shard)
    X_shuf = X_shard[idx]
    y_shuf = y_shard[idx]

    # Mini-batch SGD sul proprio shard
    for start in range(0, N_shard, BATCH_SIZE):
        X_batch = X_shuf[start:start + BATCH_SIZE]
        y_batch = y_shuf[start:start + BATCH_SIZE]

        # Gradiente locale
        grad_local = (X_batch.T @ (sigmoid(X_batch @ w) - y_batch)) / len(y_batch)

        # Allreduce: somma gradienti su tutti i rank
        comm.Allreduce(grad_local, grad_buf, op=MPI.SUM)

        # Media dei gradienti
        w -= lr * (grad_buf / size)

    epoch_time = time.time() - t_epoch

    # Valutazione ogni 10 epoche (solo rank 0)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        if rank == 0:
            loss = compute_loss(X_test, y_test, w)
            acc  = compute_accuracy(X_test, y_test, w)
            print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | acc={acc:.4f} | lr={lr:.5f} | time={epoch_time:.2f}s")
            results.append({
                "epoch":        epoch + 1,
                "loss":         round(float(loss), 6),
                "accuracy":     round(float(acc), 6),
                "lr":           round(lr, 6),
                "epoch_time_s": round(epoch_time, 4),
                "num_ranks":    size
            })

comm.Barrier()
total_time = time.time() - t_start

if rank == 0:
    print(f"\n[DONE] Tempo totale: {total_time:.2f}s")
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy", "lr", "epoch_time_s", "num_ranks"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[DONE] Risultati salvati in {RESULTS}")