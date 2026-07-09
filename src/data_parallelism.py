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
RESULTS = os.path.join(os.getenv("RESULTS_DIR"), f"data_parallelism/data_parallel__second_test_{DATASET}_np{size}.csv")

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
end_idx    = start_idx + shard_size if rank < size - 1 else N_train #last one gets all remaining samples

X_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"), dtype=np.float32).reshape(N_train, D)
y_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"), dtype=np.float32)

X_shard = X_full[start_idx:end_idx].copy()
y_shard = y_full[start_idx:end_idx].copy()
N_shard = len(y_shard)

# Validation split (5% of local shard, same convention as task_parallelism.py)
# Each rank holds its own val set; only rank 0's val metrics are logged.
VAL_SPLIT = 0.05
val_size  = max(1, int(N_shard * VAL_SPLIT))
X_val     = X_shard[:val_size].copy()
y_val     = y_shard[:val_size].copy()
X_train_s = X_shard[val_size:].copy()
y_train_s = y_shard[val_size:].copy()
N_train_s = len(y_train_s)

# Test set and output dir — rank 0 only
if rank == 0:
    X_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"),  dtype=np.float32).reshape(N_test, D)
    y_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"),  dtype=np.float32)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

if rank == 0:
    print(f"[INFO] Dataset: {DATASET} | N_train={N_train} | N_test={N_test} | D={D}")
    print(f"[INFO] Ranks: {size} | shard: ~{shard_size} | val/rank: {val_size} | "
          f"train/rank: {N_train_s} | epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")

# --- Inizializzazione pesi (identica su tutti i rank) ---
rng = np.random.default_rng(SEED)
w   = rng.normal(0, 0.01, D).astype(np.float64)

# --- Training ---
results    = []
grad_buf   = np.zeros(D, dtype=np.float64)  # buffer Allreduce

ar_window   = 0.0   # allreduce time
shuf_window = 0.0   # shuffle + fancy-index copy time

comm.Barrier()
t_start = time.time()

for epoch in range(EPOCHS):
    t_epoch    = time.time()
    t_ar_epoch = 0.0  # allreduce time accumulated across mini-batches this epoch
    lr = LR0 / (1.0 + epoch * 0.1)

    t_shuf = time.time()
    rng_local = np.random.default_rng(SEED + epoch * 100 + rank)
    idx    = rng_local.permutation(N_train_s)
    X_shuf = X_train_s[idx]   # full fancy-index copy (~1.5 GB at np=2 on epsilon)
    y_shuf = y_train_s[idx]
    shuffle_time_s = time.time() - t_shuf

    # Mini-batch SGD sul training split
    for start in range(0, N_train_s, BATCH_SIZE):
        X_batch = X_shuf[start:start + BATCH_SIZE]
        y_batch = y_shuf[start:start + BATCH_SIZE]

        # Gradiente locale
        grad_local = (X_batch.T @ (sigmoid(X_batch @ w) - y_batch)) / len(y_batch)

        # Allreduce: somma gradienti su tutti i rank — timed
        t_ar = time.time()
        comm.Allreduce(grad_local, grad_buf, op=MPI.SUM)
        t_ar_epoch += time.time() - t_ar

        # Media dei gradienti
        w -= lr * (grad_buf / size)

    epoch_time = time.time() - t_epoch
    ar_window   += t_ar_epoch
    shuf_window += shuffle_time_s

    # Evaluation every 10 epochs (rank 0 only)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        if rank == 0:
            interval       = 10 if epoch > 0 else 1
            ar_per_epoch   = ar_window   / interval
            shuf_per_epoch = shuf_window / interval

            # Test metrics (full held-out test set)
            loss = compute_loss(X_test, y_test, w)
            acc  = compute_accuracy(X_test, y_test, w)
            # Validation metrics (rank 0's local 5% val split)
            loss_val = compute_loss(X_val, y_val, w)
            acc_val  = compute_accuracy(X_val, y_val, w)

            print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | acc={acc:.4f} "
                  f"| loss_val={loss_val:.4f} | acc_val={acc_val:.4f} "
                  f"| lr={lr:.5f} | time={epoch_time:.2f}s "
                  f"| ar={ar_per_epoch:.4f}s | shuf={shuf_per_epoch:.4f}s")
            results.append({
                "epoch":            epoch + 1,
                "loss":             round(float(loss), 6),
                "accuracy":         round(float(acc), 6),
                "loss_val":         round(float(loss_val), 6),
                "acc_val":          round(float(acc_val), 6),
                "lr":               round(lr, 6),
                "epoch_time_s":     round(epoch_time, 4),
                "allreduce_time_s": round(ar_per_epoch, 4),
                "shuffle_time_s":   round(shuf_per_epoch, 4),
                "num_ranks":        size
            })
        ar_window   = 0.0   # reset windows after each reporting checkpoint
        shuf_window = 0.0

comm.Barrier()
total_time = time.time() - t_start

if rank == 0:
    print(f"\n[DONE] Tempo totale: {total_time:.2f}s")
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "loss", "accuracy", "loss_val", "acc_val",
            "lr", "epoch_time_s", "allreduce_time_s", "shuffle_time_s", "num_ranks"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"[DONE] Risultati salvati in {RESULTS}")