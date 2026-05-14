import numpy as np
import json
import os
import time
import csv
from mpi4py import MPI
from dotenv import load_dotenv

load_dotenv()

# --- MPI Init ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_workers = size - 1

DATA_DIR   = os.getenv("DATASET_DIR")
DATASET    = os.getenv("DATASET_NAME")
RESULTS    = os.path.join(os.getenv("RESULTS_DIR"), "master_worker.csv")

SEED       = int(os.getenv("SEED", 42))
EPOCHS     = int(os.getenv("EPOCHS", 50))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
LR0        = float(os.getenv("LEARNING_RATE", 0.1))

TAG_DATA   = 10
TAG_GRAD   = 20
TAG_W      = 30

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


if rank == 0:
    # =====================
    # MASTER
    # =====================
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    meta_path = os.path.join(DATA_DIR, f"{DATASET}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    N_train = meta["N_train"]
    N_test  = meta["N_test"]
    D       = meta["D"]

    print(f"[MASTER] Dataset: {DATASET} | N_train={N_train} | N_test={N_test} | D={D}")
    print(f"[MASTER] Workers: {num_workers} | epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")

    X_train = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"), dtype=np.float32).reshape(N_train, D)
    y_train = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"), dtype=np.float32)
    X_test  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"),  dtype=np.float32).reshape(N_test, D)
    y_test  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"),  dtype=np.float32)

    # Broadcast D a tutti i worker
    comm.bcast(D, root=0)

    rng = np.random.default_rng(SEED)
    w   = rng.normal(0, 0.01, D).astype(np.float64)

    results = []
    t_start = time.time()

    for epoch in range(EPOCHS):
        t_epoch = time.time()
        lr = LR0 / (1.0 + epoch * 0.1)

        # Shuffle
        idx = rng.permutation(N_train)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        # Processa mini-batch
        for start in range(0, N_train, BATCH_SIZE * num_workers):
            # Invia w a tutti i worker
            for wid in range(1, size):
                comm.Send(w, dest=wid, tag=TAG_W)

            # Distribuisci un mini-batch a ciascun worker
            for wid in range(1, size):
                b_start = start + (wid - 1) * BATCH_SIZE
                b_end   = min(b_start + BATCH_SIZE, N_train)
                if b_start >= N_train:
                    # Invia batch vuoto
                    comm.send(0, dest=wid, tag=TAG_DATA)
                    continue

                actual_size = b_end - b_start
                comm.send(actual_size, dest=wid, tag=TAG_DATA)
                comm.Send(X_shuf[b_start:b_end].copy(), dest=wid, tag=TAG_DATA + 1)
                comm.Send(y_shuf[b_start:b_end].copy(), dest=wid, tag=TAG_DATA + 2)

            # Raccogli gradienti dai worker e aggrega
            grad_sum = np.zeros(D, dtype=np.float64)
            count = 0

            for wid in range(1, size):
                has_data = comm.recv(source=wid, tag=TAG_GRAD)
                if has_data:
                    grad_local = np.empty(D, dtype=np.float64)
                    comm.Recv(grad_local, source=wid, tag=TAG_GRAD + 1)
                    grad_sum += grad_local
                    count += 1

            # Aggiorna w con media dei gradienti
            if count > 0:
                w -= lr * (grad_sum / count)

        epoch_time = time.time() - t_epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            loss = compute_loss(X_train, y_train, w)
            acc  = compute_accuracy(X_test, y_test, w)
            print(f"  Epoch {epoch+1:3d} | loss={loss:.4f} | acc={acc:.4f} | lr={lr:.5f} | time={epoch_time:.2f}s")
            results.append({
                "epoch":        epoch + 1,
                "loss":         round(float(loss), 6),
                "accuracy":     round(float(acc), 6),
                "lr":           round(lr, 6),
                "epoch_time_s": round(epoch_time, 4),
                "num_workers":  num_workers
            })

    # Segnala ai worker di terminare
    for wid in range(1, size):
        comm.Send(w, dest=wid, tag=TAG_W)
        comm.send(-1, dest=wid, tag=TAG_DATA)

    total_time = time.time() - t_start
    print(f"\n[MASTER] Tempo totale: {total_time:.2f}s")

    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "accuracy", "lr", "epoch_time_s", "num_workers"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[MASTER] Risultati salvati in {RESULTS}")

else:
    # =====================
    # WORKER
    # =====================
    D = comm.bcast(None, root=0)

    while True:
        # Ricevi w dal master
        w = np.empty(D, dtype=np.float64)
        comm.Recv(w, source=0, tag=TAG_W)

        # Ricevi dimensione batch
        batch_size = comm.recv(source=0, tag=TAG_DATA)

        if batch_size == -1:
            # Segnale di terminazione
            break

        if batch_size == 0:
            # Nessun dato — comunica che non ho gradiente
            comm.send(False, dest=0, tag=TAG_GRAD)
            continue

        # Ricevi mini-batch
        X_batch = np.empty((batch_size, D), dtype=np.float32)
        y_batch = np.empty(batch_size, dtype=np.float32)
        comm.Recv(X_batch, source=0, tag=TAG_DATA + 1)
        comm.Recv(y_batch, source=0, tag=TAG_DATA + 2)

        # Calcola gradiente locale
        grad = (X_batch.T @ (sigmoid(X_batch @ w) - y_batch)) / batch_size

        # Invia gradiente al master
        comm.send(True, dest=0, tag=TAG_GRAD)
        comm.Send(grad, dest=0, tag=TAG_GRAD + 1)