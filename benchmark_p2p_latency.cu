#include <cuda_runtime.h>
#include <cstdio>
#include <thread>
#include <atomic>

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t err = (cmd); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Simple ping-pong kernel: wait for expected value, then write response
__global__ void pingpong_kernel(volatile char* recv_buf, volatile char* send_buf, int rounds, char start_value, int device_id) {
    for (int i = 0; i < rounds; ++i) {
        char expected = start_value + i * 2 + device_id;

        // printf("0. on device %d, expected: %d, recv_buf: %d\n", device_id, expected, recv_buf[0]);
        // Wait for expected value
        while (recv_buf[0] != expected) {}

        // printf("1. on device %d, expected: %d, recv_buf: %d\n", device_id, expected, recv_buf[0]);

        // Respond with same value
        send_buf[0] = expected + 1;

        // printf("2. on device %d, expected: %d, send_buf: %d\n", device_id, expected, send_buf[0]);
    }
}

void run_gpu_thread(int device_id,
                    char* local_buf,
                    char* remote_buf,
                    int rounds,
                    char start_value,
                    cudaEvent_t start_event,
                    cudaEvent_t stop_event) {
    printf("on host\n");
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemset(local_buf, 0, 1));
    CUDA_CHECK(cudaMemset(remote_buf, 0, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_event));

    pingpong_kernel<<<1, 1>>>(local_buf, remote_buf, rounds, start_value, device_id);

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
}

int main() {
    int dev0 = 0, dev1 = 1;
    int access01 = 0, access10 = 0;

    CUDA_CHECK(cudaDeviceCanAccessPeer(&access01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access10, dev1, dev0));

    if (!access01 || !access10) {
        printf("P2P access not available between device %d and %d\n", dev0, dev1);
        return 1;
    }

    // Enable P2P
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    // Allocate cross-GPU buffers
    char *buf0;
    char *buf1;

    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMalloc(&buf0, 1));

    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaMalloc(&buf1, 1));

    // Setup timers
    cudaEvent_t start0, stop0, start1, stop1;
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaEventCreate(&start0));
    CUDA_CHECK(cudaEventCreate(&stop0));

    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    const int rounds = 100000;

    // Launch two GPU threads
    std::thread t0(run_gpu_thread, dev0, buf0, buf1, rounds, 0, start0, stop0);
    std::thread t1(run_gpu_thread, dev1, buf1, buf0, rounds, 0, start1, stop1);

    // Write first ping to GPU1's recv_buf
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaMemset(buf1, 1, 1));

    t0.join();
    t1.join();

    // Measure time
    float ms0, ms1;
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaEventElapsedTime(&ms0, start0, stop0));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start1, stop1));

    printf("GPU0: %.3f ms total, %.3f us/send\n", ms0, 1000.0f * ms0 / (rounds * 2));
    printf("GPU1: %.3f ms total, %.3f us/send\n", ms1, 1000.0f * ms1 / (rounds * 2));

    // Cleanup
    CUDA_CHECK(cudaFree(buf0));
    CUDA_CHECK(cudaFree(buf1));
    CUDA_CHECK(cudaEventDestroy(start0));
    CUDA_CHECK(cudaEventDestroy(stop0));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));

    return 0;
}
