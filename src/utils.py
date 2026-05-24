import numpy as np
import pyopencl as cl

WG_SIZE = 256

# ─────────────────────────────────────────────
# Funzioni numeriche comuni
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def compute_loss(X, y, w):
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def compute_accuracy(X, y, w):
    preds = (sigmoid(X @ w) >= 0.5).astype(np.float32)
    return np.mean(preds == y)

# ─────────────────────────────────────────────
# Task A — CPU
# ─────────────────────────────────────────────
def task_a_cpu(X_batch, y_batch, w_snapshot):
    grad = (X_batch.T @ (sigmoid(X_batch @ w_snapshot) - y_batch)) / len(y_batch)
    return grad.astype(np.float64)

# ─────────────────────────────────────────────
# Task A — GPU (OpenCL)
# ─────────────────────────────────────────────
def task_a_gpu(ctx, queue_cl, k_forward, k_gradient, X_batch, y_batch, w_snapshot):
    B, D  = X_batch.shape
    X_f32 = np.ascontiguousarray(X_batch,    dtype=np.float32)
    y_f32 = np.ascontiguousarray(y_batch,    dtype=np.float32)
    w_f32 = np.ascontiguousarray(w_snapshot, dtype=np.float32)

    mf       = cl.mem_flags
    X_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=X_f32)
    y_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=y_f32)
    w_buf    = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=w_f32)
    err_buf  = cl.Buffer(ctx, mf.READ_WRITE, size=B * np.dtype(np.float32).itemsize)
    grad_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=D * np.dtype(np.float32).itemsize)

    gs_fwd  = int(np.ceil(B / WG_SIZE)) * WG_SIZE
    gs_grad = int(np.ceil(D / WG_SIZE)) * WG_SIZE

    k_forward.set_args(X_buf, w_buf, y_buf, err_buf, np.int32(D))
    cl.enqueue_nd_range_kernel(queue_cl, k_forward,  (gs_fwd,),  (WG_SIZE,))

    k_gradient.set_args(X_buf, err_buf, grad_buf, np.int32(B), np.int32(D))
    cl.enqueue_nd_range_kernel(queue_cl, k_gradient, (gs_grad,), (WG_SIZE,))

    grad_f32 = np.empty(D, dtype=np.float32)
    cl.enqueue_copy(queue_cl, grad_f32, grad_buf)
    queue_cl.finish()
    return grad_f32.astype(np.float64)

# ─────────────────────────────────────────────
# Task B — valutazione loss su validation set
# ─────────────────────────────────────────────
def task_b(X_val, y_val, w_snapshot):
    loss = compute_loss(X_val, y_val, w_snapshot)
    acc  = compute_accuracy(X_val, y_val, w_snapshot)
    return loss, acc

# ─────────────────────────────────────────────
# Task C — preparazione prossimo mini-batch
# ─────────────────────────────────────────────
def task_c(X_train, y_train, batch_size, rng_seed):
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(y_train), size=batch_size, replace=False)
    return X_train[idx].copy(), y_train[idx].copy()

# ─────────────────────────────────────────────
# Setup OpenCL
# ─────────────────────────────────────────────
def setup_opencl(local_rank, kernel_path):
    platforms    = cl.get_platforms()
    gpu_platform = next((p for p in platforms if "NVIDIA" in p.name), platforms[0])
    device       = gpu_platform.get_devices(device_type=cl.device_type.GPU)[0]
    ctx          = cl.Context([device])
    queue_cl     = cl.CommandQueue(ctx)

    with open(kernel_path) as f:
        src = f.read()

    print(f"[rank {local_rank}] Building OpenCL program on {device.name}...",
          flush=True)
    try:
        program = cl.Program(ctx, src).build()
        print(f"[rank {local_rank}] Build OK", flush=True)
    except cl.RuntimeError as e:
        print(f"[rank {local_rank}] Build FAILED: {e}", flush=True)
        raise

    k_forward  = cl.Kernel(program, "forward_kernel")
    k_gradient = cl.Kernel(program, "gradient_kernel")

    if local_rank == 0:
        print(f"[OpenCL] Platform: {gpu_platform.name}")
        print(f"[OpenCL] Device:   {device.name}")
        print(f"[OpenCL] Max WG:   {device.max_work_group_size}")

    return ctx, queue_cl, k_forward, k_gradient