import numpy as np
import json
import os
import time
import csv
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
RESULTS = os.path.join(os.getenv("RESULTS_DIR"), f"task_parallelism/tasks_{DATASET}_np{size}.csv")

SEED       = int(os.getenv("SEED", 42))
EPOCHS     = int(os.getenv("EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LR0        = float(os.getenv("LEARNING_RATE"))
VAL_SPLIT  = 0.05  # frazione del shard per Task B



# --- Task A: calcolo gradiente ---
# w_snapshot è read-only (copia di w al momento del lancio)
def task_a(X_batch, y_batch, w_snapshot):
    grad = (X_batch.T @ (sigmoid(X_batch @ w_snapshot) - y_batch)) / len(y_batch)
    return grad.astype(np.float64)

# --- Task B: valutazione loss su validation set ---
# Legge w_snapshot (read-only) e X_val, y_val (fissi)
def task_b(X_val, y_val, w_snapshot):
    loss = compute_loss(X_val, y_val, w_snapshot)
    acc  = compute_accuracy(X_val, y_val, w_snapshot)
    return loss, acc

# --- Task C: preparazione prossimo mini-batch ---
# NON legge w — solo shuffle su indici del dataset locale
def task_c(X_train, y_train, batch_size, rng_seed):
    rng_local = np.random.default_rng(rng_seed)
    idx = rng_local.choice(len(y_train), size=batch_size, replace=False)
    return X_train[idx].copy(), y_train[idx].copy()

# --- Caricamento dataset ---
meta_path = os.path.join(DATA_DIR, f"{DATASET}_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

N_train = meta["N_train"]
N_test  = meta["N_test"]
D       = meta["D"]

# Partizionamento shard
shard_size = N_train // size
start_idx  = rank * shard_size
end_idx    = start_idx + shard_size if rank < size - 1 else N_train

X_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"), dtype=np.float32).reshape(N_train, D)
y_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"), dtype=np.float32)

X_shard = X_full[start_idx:end_idx].copy()
y_shard = y_full[start_idx:end_idx].copy()
N_shard = len(y_shard)

# Validation set locale (5% del shard) — usato da Task B
val_size  = max(1, int(N_shard * VAL_SPLIT))
X_val     = X_shard[:val_size].copy()
y_val     = y_shard[:val_size].copy()
X_train_s = X_shard[val_size:].copy()
y_train_s = y_shard[val_size:].copy()
N_train_s = len(y_train_s)

if rank == 0:
    X_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"), dtype=np.float32).reshape(N_test, D)
    y_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"), dtype=np.float32)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    print(f"[INFO] Dataset: {DATASET} | N_train={N_train} | D={D}")
    print(f"[INFO] Ranks: {size} | shard: ~{shard_size} | val_size: {val_size}")
    print(f"[INFO] epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")

# --- Inizializzazione pesi ---
rng = np.random.default_rng(SEED)
w   = rng.normal(0, 0.01, D).astype(np.float64)

# Buffer Allreduce
grad_buf = np.zeros(D, dtype=np.float64)

# Accumulatori per misura overlap
total_allreduce_time = 0.0
total_taskbc_time    = 0.0

results = []

# Pre-prepara il primo batch (step 0) prima del loop
# Task C del passo -1 — inizializzazione
rng_seed_init = SEED + rank * 10000
X_next, y_next = task_c(X_train_s, y_train_s, BATCH_SIZE, rng_seed_init)

comm.Barrier()
t_start = time.time()

with ThreadPoolExecutor(max_workers=3) as executor:

    for epoch in range(EPOCHS):
        t_epoch = time.time()
        lr = LR0 / (1.0 + epoch * 0.1)

        steps_per_epoch = N_train_s // BATCH_SIZE

        for step in range(steps_per_epoch):

            # Il batch corrente è quello preparato dal Task C del passo precedente
            X_batch = X_next
            y_batch = y_next

            # Snapshot di w — read-only per tutti i task
            w_snapshot = w.copy()

            # ── STEP 1: Task A (critico, sincrono) ──────────────────────
            # Calcola il gradiente sul batch corrente
            fut_a = executor.submit(task_a, X_batch, y_batch, w_snapshot)
            grad_local = fut_a.result()

            # ── STEP 2: Lancia Iallreduce (non-bloccante) ───────────────
            grad_buf[:] = 0.0
            t_ar0   = time.time()
            request = comm.Iallreduce(grad_local, grad_buf, op=MPI.SUM)

            # ── STEP 3: Task B e Task C in parallelo con Allreduce ───────
            # Task B: valuta loss su validation set (legge w_snapshot)
            # Task C: prepara batch per step+1 (non legge w)
            t_bc0  = time.time()
            rng_seed_next = SEED + rank * 10000 + epoch * 100000 + step + 1
            fut_b  = executor.submit(task_b, X_val, y_val, w_snapshot)
            fut_c  = executor.submit(task_c, X_train_s, y_train_s,
                                     BATCH_SIZE, rng_seed_next)

            loss_val, acc_val = fut_b.result()
            X_next, y_next    = fut_c.result()  # batch pronto per il prossimo step
            t_bc1  = time.time()

            # ── STEP 4: Attendi Allreduce ────────────────────────────────
            request.Wait()
            t_ar1 = time.time()

            # ── STEP 5: Aggiorna w ───────────────────────────────────────
            w -= lr * (grad_buf / size)

            # Accumula tempi
            total_allreduce_time += (t_ar1 - t_ar0)
            total_taskbc_time    += (t_bc1 - t_bc0)

        epoch_time = time.time() - t_epoch

        # Sostituisci il blocco di valutazione ogni 10 epoche con questo:
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if rank == 0:
                loss_test = compute_loss(X_test, y_test, w)
                acc_test  = compute_accuracy(X_test, y_test, w)
                
                # Numero di epoche nell'intervallo corrente
                interval = 10 if epoch > 0 else 1
                
                ar_per_epoch = total_allreduce_time / interval
                bc_per_epoch = total_taskbc_time    / interval
                overlap      = min(ar_per_epoch, bc_per_epoch)

                print(f"  Epoch {epoch+1:3d} | loss={loss_test:.4f} | acc={acc_test:.4f} "
                    f"| lr={lr:.5f} | time={epoch_time:.2f}s "
                    f"| ar={ar_per_epoch:.4f}s "
                    f"| bc={bc_per_epoch:.4f}s "
                    f"| overlap={overlap:.4f}s")
                results.append({
                    "epoch":            epoch + 1,
                    "loss":             round(float(loss_test), 6),
                    "accuracy":         round(float(acc_test), 6),
                    "lr":               round(lr, 6),
                    "epoch_time_s":     round(epoch_time, 4),
                    "allreduce_time_s": round(ar_per_epoch, 4),
                    "taskbc_time_s":    round(bc_per_epoch, 4),
                    "overlap_s":        round(overlap, 4),
                    "num_ranks":        size
                })
            # Reset accumulatori dopo ogni finestra
            total_allreduce_time = 0.0
            total_taskbc_time    = 0.0

comm.Barrier()
total_time = time.time() - t_start

if rank == 0:
    print(f"\n[DONE] Tempo totale: {total_time:.2f}s")
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "loss", "accuracy", "lr",
            "epoch_time_s", "allreduce_time_s", "taskbc_time_s",
            "overlap_s", "num_ranks"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"[DONE] Risultati salvati in {RESULTS}")