/*
 * Kernel: gradient_kernel
 * Ogni work-item gestisce una feature d (colonna di X).
 * Calcola il contributo al gradiente: g[d] = sum_i (err[i] * X[i,d]) / B
 * con bounds guard per global_size > D.
 */
__kernel void gradient_kernel(
    __global const float* X,      // [B x D] row-major
    __global const float* err,    // [B] = sigmoid(X@w) - y
    __global       float* grad,   // [D] output gradiente
    const int B,                  // batch size
    const int D                   // num features
) {
    int d = get_global_id(0);
    if (d >= D) return;           // bounds guard

    float g = 0.0f;
    for (int i = 0; i < B; i++) {
        g += err[i] * X[i * D + d];
    }
    grad[d] = g / (float)B;
}

/*
 * Kernel: forward_kernel
 * Ogni work-item gestisce un sample i (riga di X).
 * Calcola err[i] = sigmoid(X[i,:] @ w) - y[i]
 */
__kernel void forward_kernel(
    __global const float* X,      // [B x D] row-major
    __global const float* w,      // [D]
    __global const float* y,      // [B]
    __global       float* err,    // [B] output
    const int D
) {
    int i = get_global_id(0);

    float z = 0.0f;
    for (int d = 0; d < D; d++) {
        z += X[i * D + d] * w[d];
    }
    // sigmoid con clamp per stabilità numerica
    float sig;
    if (z >= 0.0f) {
        sig = 1.0f / (1.0f + exp(-z));
    } else {
        float ez = exp(z);
        sig = ez / (1.0f + ez);
    }
    err[i] = sig - y[i];
}