import numpy as np
import json
import os
import time
import csv
import pyopencl as cl
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from dotenv import load_dotenv
from utils import (sigmoid, compute_loss, compute_accuracy,
                   task_a_cpu, task_a_gpu, task_b, task_c, setup_opencl)

load_dotenv("/home/mpiuser/test/src/.env")

# ─────────────────────────────────────────────
# MPI Init globale
# ─────────────────────────────────────────────
comm       = MPI.COMM_WORLD
rank       = comm.Get_rank()
print(f"[rank {rank}] MPI init OK", flush=True)
world_size = comm.Get_size()

MASTER_RANK     = 0
RANKS_PER_GROUP = 3
NUM_GROUPS      = (world_size - 1) // RANKS_PER_GROUP

# ─────────────────────────────────────────────
# Configurazione
# ─────────────────────────────────────────────
DATA_DIR    = os.getenv("DATASET_DIR")
DATASET     = os.getenv("DATASET_NAME", "epsilon_normalized")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/home/mpiuser/test/results")
KERNEL_PATH = os.getenv("KERNEL_DIR", "/home/mpiuser/test/kernels") + "/gradient.cl"
EPOCHS      = int(os.getenv("EPOCHS", 50))
SEED        = int(os.getenv("SEED", 42))
VAL_SPLIT   = 0.05

if rank == 0:
    print(f"DATA_DIR={DATA_DIR}", flush=True)
    print(f"DATASET={DATASET}", flush=True)
    print(f"RESULTS_DIR={RESULTS_DIR}", flush=True)
    print(f"world_size={world_size} | NUM_GROUPS={NUM_GROUPS} | "
          f"ranks usati={1 + NUM_GROUPS * RANKS_PER_GROUP}", flush=True)

# ─────────────────────────────────────────────
# comm.Split — chiamata collettiva su TUTTI i rank
# ─────────────────────────────────────────────
if rank == MASTER_RANK:
    group_comm = comm.Split(MPI.UNDEFINED, rank)
else:
    group_id = (rank - 1) // RANKS_PER_GROUP
    if group_id >= NUM_GROUPS:
        group_comm = comm.Split(MPI.UNDEFINED, rank)
    else:
        group_comm = comm.Split(group_id, rank)
print(f"[rank {rank}] Split done", flush=True)

# ─────────────────────────────────────────────
# Job queue
# ─────────────────────────────────────────────
CONFIGS = [
    {"lr": 0.16,  "batch": 4096,  "use_gpu": False},
    {"lr": 0.64,  "batch": 16384, "use_gpu": False},
    {"lr": 0.16,  "batch": 4096,  "use_gpu": True},
    {"lr": 0.64,  "batch": 16384, "use_gpu": True},
]

TAG_JOB    = 1
TAG_RESULT = 2

# ─────────────────────────────────────────────
# Training completo (L2+L3 o L2+L3+L4)
# ─────────────────────────────────────────────
def run_training(group_comm, local_rank, group_id,
                 lr, batch_size, use_gpu, dataset):

    gsize = group_comm.Get_size()
    label = f"G{group_id} | {'GPU' if use_gpu else 'CPU'} | lr={lr} | B={batch_size}"

    print(f"[{label}] rank={rank} local_rank={local_rank} loading meta...", flush=True)

    meta_path = os.path.join(DATA_DIR, f"{dataset}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    N_train = meta["N_train"]
    N_test  = meta["N_test"]
    D       = meta["D"]

    print(f"[{label}] rank={rank} meta OK — N={N_train} D={D}", flush=True)

    shard_size = N_train // gsize
    start_idx  = local_rank * shard_size
    end_idx    = start_idx + shard_size if local_rank < gsize - 1 else N_train

    print(f"[{label}] rank={rank} loading X_train shard [{start_idx}:{end_idx}]...", flush=True)

    X_full = np.fromfile(os.path.join(DATA_DIR, f"{dataset}_X_train.bin"),
                         dtype=np.float32).reshape(N_train, D)
    y_full = np.fromfile(os.path.join(DATA_DIR, f"{dataset}_y_train.bin"),
                         dtype=np.float32)

    print(f"[{label}] rank={rank} dataset loaded", flush=True)

    X_shard   = X_full[start_idx:end_idx].copy()
    y_shard   = y_full[start_idx:end_idx].copy()
    N_shard   = len(y_shard)
    val_size  = max(1, int(N_shard * VAL_SPLIT))
    X_val     = X_shard[:val_size].copy()
    y_val     = y_shard[:val_size].copy()
    X_train_s = X_shard[val_size:].copy()
    y_train_s = y_shard[val_size:].copy()
    N_train_s = len(y_train_s)

    X_test = None
    y_test = None
    if local_rank == 0:
        X_test = np.fromfile(os.path.join(DATA_DIR, f"{dataset}_X_test.bin"),
                             dtype=np.float32).reshape(N_test, D)
        y_test = np.fromfile(os.path.join(DATA_DIR, f"{dataset}_y_test.bin"),
                             dtype=np.float32)
        print(f"[{label}] START — N={N_train} D={D} epochs={EPOCHS}", flush=True)

    if use_gpu:
        print(f"[{label}] rank={rank} setting up OpenCL...", flush=True)
        ctx, queue_cl, k_forward, k_gradient = setup_opencl(local_rank, KERNEL_PATH)
        print(f"[{label}] rank={rank} OpenCL OK", flush=True)

    rng          = np.random.default_rng(SEED)
    w            = rng.normal(0, 0.01, D).astype(np.float64)
    grad_buf_mpi = np.zeros(D, dtype=np.float64)

    total_ar  = 0.0
    total_bc  = 0.0
    total_gpu = 0.0

    last_loss_val = 0.0
    last_acc_val  = 0.0
    epoch_results = []

    X_next, y_next = task_c(X_train_s, y_train_s, batch_size,
                             SEED + local_rank * 10000)

    print(f"[{label}] rank={rank} reaching Barrier...", flush=True)
    group_comm.Barrier()
    print(f"[{label}] rank={rank} Barrier passed, training start", flush=True)

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        for epoch in range(EPOCHS):
            t_epoch = time.time()
            lr_eff  = lr / (1.0 + epoch * 0.1)
            steps   = N_train_s // batch_size

            for step in range(steps):
                X_batch    = X_next
                y_batch    = y_next
                w_snapshot = w.copy()

                # Debug solo step 0 epoca 0
                dbg = (step == 0 and epoch == 0 and local_rank == 0)

                if dbg:
                    print(f"[{label}] Step 0 — Task A start", flush=True)

                t_a0 = time.time()
                if use_gpu:
                    fut_a = executor.submit(task_a_gpu, ctx, queue_cl,
                                            k_forward, k_gradient,
                                            X_batch, y_batch, w_snapshot)
                else:
                    fut_a = executor.submit(task_a_cpu,
                                            X_batch, y_batch, w_snapshot)
                grad_local = fut_a.result()
                t_a1 = time.time()

                if dbg:
                    print(f"[{label}] Step 0 — Task A done ({t_a1-t_a0:.3f}s), "
                          f"launching Iallreduce", flush=True)

                grad_buf_mpi[:] = 0.0
                t_ar0   = time.time()
                request = group_comm.Iallreduce(grad_local, grad_buf_mpi,
                                                op=MPI.SUM)

                if dbg:
                    print(f"[{label}] Step 0 — Iallreduce launched, "
                          f"starting Task B+C", flush=True)

                t_bc0         = time.time()
                rng_seed_next = SEED + local_rank*10000 + epoch*100000 + step+1
                fut_b = executor.submit(task_b, X_val, y_val, w_snapshot)
                fut_c = executor.submit(task_c, X_train_s, y_train_s,
                                        batch_size, rng_seed_next)

                last_loss_val, last_acc_val = fut_b.result()
                X_next, y_next              = fut_c.result()
                t_bc1 = time.time()

                if dbg:
                    print(f"[{label}] Step 0 — Task B+C done ({t_bc1-t_bc0:.3f}s), "
                          f"waiting Allreduce...", flush=True)

                request.Wait()
                t_ar1 = time.time()

                if dbg:
                    print(f"[{label}] Step 0 — Allreduce done ({t_ar1-t_ar0:.3f}s), "
                          f"updating w", flush=True)

                w -= lr_eff * (grad_buf_mpi / gsize)

                if dbg:
                    print(f"[{label}] Step 0 COMPLETE", flush=True)

                total_gpu += (t_a1  - t_a0)
                total_ar  += (t_ar1 - t_ar0)
                total_bc  += (t_bc1 - t_bc0)

            epoch_time = time.time() - t_epoch

            if (epoch + 1) % 10 == 0 or epoch == 0:
                if local_rank == 0:
                    loss_t = compute_loss(X_test, y_test, w)
                    acc_t  = compute_accuracy(X_test, y_test, w)
                    intv   = 10 if epoch > 0 else 1
                    print(f"  [{label}] Epoch {epoch+1:3d} "
                          f"| loss={loss_t:.4f} | acc={acc_t:.4f} "
                          f"| time={epoch_time:.2f}s "
                          f"| gpu={total_gpu/intv:.4f}s "
                          f"| ar={total_ar/intv:.4f}s "
                          f"| overlap={min(total_ar,total_bc)/intv:.4f}s",
                          flush=True)
                    epoch_results.append({
                        "epoch":        epoch + 1,
                        "loss":         round(float(loss_t), 6),
                        "accuracy":     round(float(acc_t), 6),
                        "loss_val":     round(float(last_loss_val), 6),
                        "acc_val":      round(float(last_acc_val), 6),
                        "lr":           round(lr_eff, 6),
                        "epoch_time_s": round(epoch_time, 4),
                        "gpu_time_s":   round(total_gpu / intv, 4),
                        "ar_time_s":    round(total_ar  / intv, 4),
                        "bc_time_s":    round(total_bc  / intv, 4),
                        "overlap_s":    round(min(total_ar, total_bc) / intv, 4),
                    })
                total_gpu = 0.0
                total_ar  = 0.0
                total_bc  = 0.0

    group_comm.Barrier()
    total_time = time.time() - t_start

    if local_rank == 0:
        mode  = "gpu" if use_gpu else "cpu"
        fname = (f"job_{dataset}_{mode}"
                 f"_lr{lr:.4f}_b{batch_size}"
                 f"_g{group_id}.csv")
        fpath = os.path.join(RESULTS_DIR, fname)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(fpath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_results[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_results)
        print(f"  [{label}] DONE — time={total_time:.1f}s | "
              f"final_acc={epoch_results[-1]['accuracy']:.4f} | "
              f"saved: {fname}", flush=True)

        return {
            "lr":       lr,
            "batch":    batch_size,
            "use_gpu":  use_gpu,
            "dataset":  dataset,
            "accuracy": epoch_results[-1]["accuracy"],
            "loss":     epoch_results[-1]["loss"],
            "time":     round(total_time, 2),
            "file":     fname,
        }
    return None

# ═════════════════════════════════════════════
# MASTER
# ═════════════════════════════════════════════
if rank == MASTER_RANK:
    group_leaders = [1 + i * RANKS_PER_GROUP for i in range(NUM_GROUPS)]
    queue   = list(CONFIGS)
    active  = {}
    results = []

    print(f"\n{'='*60}")
    print(f"  HPC Job Queue — {len(queue)} configs | {NUM_GROUPS} groups")
    print(f"  Dataset: {DATASET} | Epochs: {EPOCHS}")
    print(f"{'='*60}\n", flush=True)

    for leader in group_leaders:
        if queue:
            cfg = queue.pop(0)
            comm.send(cfg, dest=leader, tag=TAG_JOB)
            active[leader] = cfg
            print(f"  → Sent to G{(leader-1)//RANKS_PER_GROUP}: "
                  f"lr={cfg['lr']} B={cfg['batch']} "
                  f"{'GPU' if cfg['use_gpu'] else 'CPU'}", flush=True)
        else:
            comm.send(None, dest=leader, tag=TAG_JOB)

    while active:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        leader = status.Get_source()
        results.append(result)
        del active[leader]

        print(f"  ← Result from G{(leader-1)//RANKS_PER_GROUP}: "
              f"acc={result['accuracy']:.4f} | "
              f"lr={result['lr']} B={result['batch']} "
              f"{'GPU' if result['use_gpu'] else 'CPU'}", flush=True)

        if queue:
            cfg = queue.pop(0)
            comm.send(cfg, dest=leader, tag=TAG_JOB)
            active[leader] = cfg
            print(f"  → Sent to G{(leader-1)//RANKS_PER_GROUP}: "
                  f"lr={cfg['lr']} B={cfg['batch']} "
                  f"{'GPU' if cfg['use_gpu'] else 'CPU'}", flush=True)
        else:
            comm.send(None, dest=leader, tag=TAG_JOB)

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    print(f"\n{'='*60}")
    print("  HYPERPARAMETER SEARCH — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  {'LR':>8} {'Batch':>7} {'Mode':>5} "
          f"{'Accuracy':>10} {'Loss':>8} {'Time':>8}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['lr']:>8.4f} {r['batch']:>7d} "
              f"{'GPU' if r['use_gpu'] else 'CPU':>5} "
              f"{r['accuracy']:>10.4f} {r['loss']:>8.4f} "
              f"{r['time']:>7.1f}s")
    print(f"\n  Best: lr={results[0]['lr']} | "
          f"batch={results[0]['batch']} | "
          f"{'GPU' if results[0]['use_gpu'] else 'CPU'} | "
          f"acc={results[0]['accuracy']:.4f}")
    print(f"{'='*60}\n")

    summary_path = os.path.join(RESULTS_DIR, f"job_summary_{DATASET}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "batch", "use_gpu",
                                                "dataset", "accuracy",
                                                "loss", "time", "file"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Summary saved: {summary_path}", flush=True)

# ═════════════════════════════════════════════
# WORKER GROUPS
# ═════════════════════════════════════════════
else:
    group_id = (rank - 1) // RANKS_PER_GROUP

    if group_id < NUM_GROUPS:
        local_rank = group_comm.Get_rank()
        is_leader  = (local_rank == 0)

        print(f"[rank {rank}] group_id={group_id} local_rank={local_rank} "
              f"is_leader={is_leader} — waiting for job", flush=True)

        while True:
            if is_leader:
                cfg = comm.recv(source=MASTER_RANK, tag=TAG_JOB)
            else:
                cfg = None

            print(f"[rank {rank}] bcast cfg...", flush=True)
            cfg = group_comm.bcast(cfg, root=0)
            print(f"[rank {rank}] bcast done, cfg={cfg}", flush=True)

            if cfg is None:
                print(f"[rank {rank}] received None — terminating", flush=True)
                break

            if is_leader:
                print(f"[G{group_id} | rank {rank}] Job received: "
                      f"lr={cfg['lr']} B={cfg['batch']} "
                      f"{'GPU' if cfg['use_gpu'] else 'CPU'} — loading...",
                      flush=True)

            result = run_training(
                group_comm = group_comm,
                local_rank = local_rank,
                group_id   = group_id,
                lr         = cfg["lr"],
                batch_size = cfg["batch"],
                use_gpu    = cfg["use_gpu"],
                dataset    = DATASET,
            )

            if is_leader:
                print(f"[rank {rank}] sending result to master", flush=True)
                comm.send(result, dest=MASTER_RANK, tag=TAG_RESULT)
    else:
        print(f"[rank {rank}] excess rank — idle", flush=True)