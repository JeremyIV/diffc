#include <curand_kernel.h>

extern "C" {

__global__ void generate_sample_kernel(
        int dim,
        unsigned long long shared_seed,
        unsigned long long idx,
        float* sample_out) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState state;
        curand_init(shared_seed, 0, idx * dim, &state);
        //curand_init(shared_seed + idx, 0, 0, &state);

        for (int i = 0; i < dim; i++) {
            sample_out[i] = curand_normal(&state);
        }
    }
}

__global__ void reverse_channel_encode_kernel(
    const float* mu_q,
    int dim,
    unsigned long long K,
    unsigned long long shared_seed,
    float* log_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    curandState state;
    // Cast before multiplying: idx * dim is an int32 product, which wraps when
    // idx * dim >= 2^31 (e.g. the upper candidates of a single 2^16-candidate
    // chunk over >= 32768 dims). generate_sample_kernel computes the same
    // offset in 64 bits, so the sample scored here would silently differ from
    // the sample regenerated for the winning index.
    curand_init(shared_seed, 0, ((unsigned long long)idx) * dim, &state);

    //curand_init(shared_seed + idx, 0, 0, &state);
    
    float log_w_value = 0.0f;
    for (int i = 0; i < dim; i++) {
        float sample_value = curand_normal(&state);
        //log_w_value += 0.5 * (sample_value * sample_value - (sample_value - mu_q[i]) * (sample_value - mu_q[i]));
        log_w_value += sample_value*mu_q[i];
    }

    log_w[idx] = log_w_value;
}

} // extern "C"