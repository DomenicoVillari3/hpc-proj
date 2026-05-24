import numpy as np
import json
import os
import time
import csv
import pyopencl as cl
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from dotenv import load_dotenv
from utils import *

load_dotenv()

# --- MPI Init ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

DATA_DIR    = os.getenv("DATASET_DIR")
DATASET     = os.getenv("DATASET_NAME")
RESULTS = os.path.join(os.getenv("RESULTS_DIR"), f"opencl/opencl_{DATASET}_np{size}.csv")
KERNEL_PATH = os.path.join(os.getenv("KERNEL_DIR",
              "/home/mpiuser/test/kernels"), "gradient.cl")

SEED        = int(os.getenv("SEED", 42))
EPOCHS      = int(os.getenv("EPOCHS", 50))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 4096))
LR0         = float(os.getenv("LEARNING_RATE", 0.1))
VAL_SPLIT   = 0.05
WG_SIZE     = 256

# --- Setup OpenCL ---
def setup_opencl():
    platforms = cl.get_platforms()
    gpu_platform = None
    for p in platforms:
        if "NVIDIA" in p.name or "nvidia" in p.name.lower():
            gpu_platform = p
            break
    if gpu_platform is None:
        gpu_platform = platforms[0]

    devices = gpu_platform.get_devices(device_type=cl.device_type.GPU)
    device  = devices[0]
    ctx     = cl.Context([device])
    queue   = cl.CommandQueue(ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)

    print(f"[rank {rank}] Building OpenCL program on {device.name}...", flush=True)
    with open(KERNEL_PATH, "r") as f:
        src = f.read()
    try:
        program = cl.Program(ctx, src).build()
        print(f"[rank {rank}] Build OK", flush=True)
    except cl.RuntimeError as e:
        print(f"[rank {rank}] Build FAILED: {e}", flush=True)
        raise

    k_forward  = cl.Kernel(program, "forward_kernel")
    k_gradient = cl.Kernel(program, "gradient_kernel")

    if rank == 0:
        print(f"[OpenCL] Platform: {gpu_platform.name}")
        print(f"[OpenCL] Device:   {device.name}")
        print(f"[OpenCL] Max WG:   {device.max_work_group_size}")

    return ctx, queue, k_forward, k_gradient

# --- Task A: gradiente su GPU ---
def task_a_gpu(ctx, queue, k_forward, k_gradient, X_batch, y_batch, w_snapshot):
    B, D = X_batch.shape

    X_f32 = np.ascontiguousarray(X_batch, dtype=np.float32)
    y_f32 = np.ascontiguousarray(y_batch, dtype=np.float32)
    w_f32 = np.ascontiguousarray(w_snapshot, dtype=np.float32)

    mf = cl.mem_flags
    X_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=X_f32)
    y_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=y_f32)
    w_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=w_f32)
    err_buf  = cl.Buffer(ctx, mf.READ_WRITE, size=B * np.dtype(np.float32).itemsize)
    grad_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=D * np.dtype(np.float32).itemsize)

    gs_forward = int(np.ceil(B / WG_SIZE)) * WG_SIZE
    k_forward.set_args(X_buf, w_buf, y_buf, err_buf, np.int32(D))
    cl.enqueue_nd_range_kernel(queue, k_forward, (gs_forward,), (WG_SIZE,))

    gs_grad = int(np.ceil(D / WG_SIZE)) * WG_SIZE
    k_gradient.set_args(X_buf, err_buf, grad_buf, np.int32(B), np.int32(D))
    cl.enqueue_nd_range_kernel(queue, k_gradient, (gs_grad,), (WG_SIZE,))

    grad_f32 = np.empty(D, dtype=np.float32)
    cl.enqueue_copy(queue, grad_f32, grad_buf)
    queue.finish()

    return grad_f32.astype(np.float64)

# --- Task B: valutazione loss su validation set (CPU) ---
def task_b(X_val, y_val, w_snapshot):
    loss = compute_loss(X_val, y_val, w_snapshot)
    acc  = compute_accuracy(X_val, y_val, w_snapshot)
    return loss, acc

# --- Task C: preparazione prossimo mini-batch (CPU) ---
def task_c(X_train, y_train, batch_size, rng_seed):
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(y_train), size=batch_size, replace=False)
    return X_train[idx].copy(), y_train[idx].copy()

# --- Caricamento dataset ---
meta_path = os.path.join(DATA_DIR, f"{DATASET}_meta.json")
with open(meta_path) as f:
    meta = json.load(f)

N_train = meta["N_train"]
N_test  = meta["N_test"]
D       = meta["D"]

shard_size = N_train // size
start_idx  = rank * shard_size
end_idx    = start_idx + shard_size if rank < size - 1 else N_train

X_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_train.bin"),
                      dtype=np.float32).reshape(N_train, D)
y_full  = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_train.bin"),
                      dtype=np.float32)

X_shard = X_full[start_idx:end_idx].copy()
y_shard = y_full[start_idx:end_idx].copy()
N_shard = len(y_shard)

val_size  = max(1, int(N_shard * VAL_SPLIT))
X_val     = X_shard[:val_size].copy()
y_val     = y_shard[:val_size].copy()
X_train_s = X_shard[val_size:].copy()
y_train_s = y_shard[val_size:].copy()
N_train_s = len(y_train_s)

if rank == 0:
    X_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_X_test.bin"),
                         dtype=np.float32).reshape(N_test, D)
    y_test = np.fromfile(os.path.join(DATA_DIR, f"{DATASET}_y_test.bin"),
                         dtype=np.float32)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    print(f"[INFO] Dataset: {DATASET} | N_train={N_train} | D={D}")
    print(f"[INFO] Ranks: {size} | shard: ~{shard_size} | val_size: {val_size}")
    print(f"[INFO] epochs={EPOCHS} | batch={BATCH_SIZE} | lr0={LR0}")

# --- Setup OpenCL ---
ctx, queue, k_forward, k_gradient = setup_opencl()

# --- Inizializzazione pesi ---
rng = np.random.default_rng(SEED)
w   = rng.normal(0, 0.01, D).astype(np.float64)

# Buffer Allreduce MPI
grad_buf_mpi = np.zeros(D, dtype=np.float64)

# Accumulatori timing
total_allreduce_time = 0.0
total_taskbc_time    = 0.0
total_gpu_time       = 0.0

results        = []
last_loss_val  = 0.0
last_acc_val   = 0.0

# Pre-prepara primo batch
X_next, y_next = task_c(X_train_s, y_train_s, BATCH_SIZE,
                         SEED + rank * 10000)

comm.Barrier()
t_start = time.time()

with ThreadPoolExecutor(max_workers=3) as executor:

    for epoch in range(EPOCHS):
        t_epoch = time.time()
        lr = LR0 / (1.0 + epoch * 0.1)

        steps_per_epoch = N_train_s // BATCH_SIZE

        for step in range(steps_per_epoch):

            X_batch    = X_next
            y_batch    = y_next
            w_snapshot = w.copy()

            # STEP 1: Task A su GPU (sincrono)
            t_gpu0     = time.time()
            fut_a      = executor.submit(task_a_gpu, ctx, queue,
                                         k_forward, k_gradient,
                                         X_batch, y_batch, w_snapshot)
            grad_local = fut_a.result()
            t_gpu1     = time.time()

            # STEP 2: Iallreduce non-bloccante
            grad_buf_mpi[:] = 0.0
            t_ar0   = time.time()
            request = comm.Iallreduce(grad_local, grad_buf_mpi, op=MPI.SUM)

            # STEP 3: Task B e Task C in parallelo con Allreduce
            t_bc0         = time.time()
            rng_seed_next = SEED + rank * 10000 + epoch * 100000 + step + 1
            fut_b         = executor.submit(task_b, X_val, y_val, w_snapshot)
            fut_c         = executor.submit(task_c, X_train_s, y_train_s,
                                            BATCH_SIZE, rng_seed_next)

            last_loss_val, last_acc_val = fut_b.result()
            X_next, y_next              = fut_c.result()
            t_bc1 = time.time()

            # STEP 4: Attendi Allreduce
            request.Wait()
            t_ar1 = time.time()

            # STEP 5: Aggiorna w
            w -= lr * (grad_buf_mpi / size)

            total_gpu_time       += (t_gpu1 - t_gpu0)
            total_allreduce_time += (t_ar1 - t_ar0)
            total_taskbc_time    += (t_bc1 - t_bc0)

        epoch_time = time.time() - t_epoch

        if (epoch + 1) % 10 == 0 or epoch == 0:
            if rank == 0:
                loss_test = compute_loss(X_test, y_test, w)
                acc_test  = compute_accuracy(X_test, y_test, w)

                interval      = 10 if epoch > 0 else 1
                ar_per_epoch  = total_allreduce_time / interval
                bc_per_epoch  = total_taskbc_time    / interval
                gpu_per_epoch = total_gpu_time       / interval
                overlap       = min(ar_per_epoch, bc_per_epoch)

                print(f"  Epoch {epoch+1:3d} | loss={loss_test:.4f} "
                      f"| acc={acc_test:.4f} "
                      f"| loss_val={last_loss_val:.4f} "
                      f"| acc_val={last_acc_val:.4f} "
                      f"| lr={lr:.5f} | time={epoch_time:.2f}s "
                      f"| gpu={gpu_per_epoch:.4f}s "
                      f"| ar={ar_per_epoch:.4f}s "
                      f"| bc={bc_per_epoch:.4f}s "
                      f"| overlap={overlap:.4f}s")
                results.append({
                    "epoch":            epoch + 1,
                    "loss":             round(float(loss_test), 6),
                    "accuracy":         round(float(acc_test), 6),
                    "loss_val":         round(float(last_loss_val), 6),
                    "acc_val":          round(float(last_acc_val), 6),
                    "lr":               round(lr, 6),
                    "epoch_time_s":     round(epoch_time, 4),
                    "gpu_time_s":       round(gpu_per_epoch, 4),
                    "allreduce_time_s": round(ar_per_epoch, 4),
                    "taskbc_time_s":    round(bc_per_epoch, 4),
                    "overlap_s":        round(overlap, 4),
                    "num_ranks":        size
                })

            total_gpu_time       = 0.0
            total_allreduce_time = 0.0
            total_taskbc_time    = 0.0

comm.Barrier()
total_time = time.time() - t_start

if rank == 0:
    print(f"\n[DONE] Tempo totale: {total_time:.2f}s")
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "loss", "accuracy", "loss_val", "acc_val",
            "lr", "epoch_time_s", "gpu_time_s",
            "allreduce_time_s", "taskbc_time_s", "overlap_s", "num_ranks"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"[DONE] Risultati salvati in {RESULTS}")