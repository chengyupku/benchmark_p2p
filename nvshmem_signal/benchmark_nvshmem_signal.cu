#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 1

__global__ void benchmark_signal_kernel(uint64_t* signal_buf, int mype, int npes, int rounds, uint64_t start_value) {
    int peer = (mype + 1) % npes;
    if (threadIdx.x == 0) {
        for (int i = 0; i < rounds; ++i) {
            uint64_t expected = start_value + i * 2 + mype;
            nvshmem_signal_wait_until(signal_buf, NVSHMEM_CMP_EQ, expected);
            nvshmemx_signal_op(signal_buf, uint64_t(i * 2 + mype + 1), NVSHMEM_SIGNAL_SET, peer);
        }
    }
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    uint64_t *signal_buf;

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    signal_buf = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    assert(signal_buf != NULL);

    int rounds = 10;
    uint64_t start_value = 0;

    cudaEvent_t start, stop;
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    // ------------------------------

    benchmark_signal_kernel<<<1, THREADS_PER_BLOCK>>>(signal_buf, mype, npes, rounds, start_value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("GPU%d: %.3f ms total, %.3f us/send\n", mype, elapsed_ms, 1000.0f * elapsed_ms / (rounds * 2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // ------------------------------

    nvshmem_free(signal_buf);
    nvshmem_finalize();

    return 0;
}