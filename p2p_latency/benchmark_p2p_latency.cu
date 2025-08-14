#include <cuda_runtime.h>
#include <cstdio>
#include <thread>
#include <atomic>
#include <vector>
#include <iomanip>
#include <iostream>
#include <algorithm>

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t err = (cmd); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Simple ping-pong kernel: wait for expected value, then write response
__global__ void pingpong_kernel(volatile int* recv_buf, volatile int* send_buf, int rounds, int start_value, int binary_id) {
    // printf("on device %d\n", binary_id);
    for (int i = 0; i < rounds; ++i) {
        int expected = start_value + i * 2 + binary_id;

        // printf("0. on device %d, expected: %d, recv_buf: %d\n", binary_id, expected, recv_buf[0]);
        // Wait for expected value
        int value = -1;
        while (value != expected) {
            asm volatile ("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(recv_buf));
        }

        // printf("1. on device %d, expected: %d, recv_buf: %d\n", binary_id, expected, recv_buf[0]);

        // Respond with same value
        send_buf[0] = expected + 1;

        // printf("2. on device %d, expected: %d, send_buf: %d\n", binary_id, expected, send_buf[0]);
    }
}

void run_gpu_thread(int device_id,
                    int binary_id,
                    int* local_buf,
                    int* remote_buf,
                    int rounds,
                    int start_value,
                    cudaEvent_t start_event,
                    cudaEvent_t stop_event) {
    printf("on host\n");
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemset(local_buf, 0, 1));
    CUDA_CHECK(cudaMemset(remote_buf, 0, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start_event));

    pingpong_kernel<<<1, 1>>>(local_buf, remote_buf, rounds, start_value, binary_id);

    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
}

float profile(int dev0, int dev1) {
    int access01 = 0, access10 = 0;

    CUDA_CHECK(cudaDeviceCanAccessPeer(&access01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access10, dev1, dev0));

    if (!access01 || !access10) {
        printf("P2P access not available between device %d and %d\n", dev0, dev1);
        return -1.0f; // Return -1 to indicate no access
    }

    // // Enable P2P
    // CUDA_CHECK(cudaSetDevice(dev0));
    // CUDA_CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    // CUDA_CHECK(cudaSetDevice(dev1));
    // CUDA_CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    // Allocate cross-GPU buffers
    int *buf0;
    int *buf1;

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
    std::thread t0(run_gpu_thread, dev0, dev0 < dev1 ? 0 : 1, buf0, buf1, rounds, 0, start0, stop0);
    std::thread t1(run_gpu_thread, dev1, dev1 < dev0 ? 0 : 1, buf1, buf0, rounds, 0, start1, stop1);

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

    // printf("GPU %d: %.3f ms total, %.3f us/send\n", dev0, ms0, 1000.0f * ms0 / (rounds * 2));
    // printf("GPU %d: %.3f ms total, %.3f us/send\n", dev1, ms1, 1000.0f * ms1 / (rounds * 2));

    // Cleanup
    CUDA_CHECK(cudaFree(buf0));
    CUDA_CHECK(cudaFree(buf1));
    CUDA_CHECK(cudaEventDestroy(start0));
    CUDA_CHECK(cudaEventDestroy(stop0));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));

    return 1000.0f * ((ms0 + ms1) / 2.0f) / (rounds * 2);
}

void print_latency_table(const std::vector<std::vector<float>>& latency_matrix, int device_count) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    P2P Latency Table (microseconds)                    " << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Print header row
    std::cout << std::setw(8) << "Device";
    for (int j = 0; j < device_count; ++j) {
        std::cout << std::setw(10) << j;
    }
    std::cout << std::endl;
    
    // Print separator line
    std::cout << std::string(8 + device_count * 10, '-') << std::endl;
    
    // Print data rows
    for (int i = 0; i < device_count; ++i) {
        std::cout << std::setw(8) << i;
        for (int j = 0; j < device_count; ++j) {
            if (i == j) {
                std::cout << std::setw(10) << "N/A";
            } else if (latency_matrix[i][j] < 0) {
                std::cout << std::setw(10) << "N/A";
            } else {
                std::cout << std::setw(10) << std::fixed << std::setprecision(3) << latency_matrix[i][j];
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << std::string(80, '=') << std::endl;
}

void print_summary_stats(const std::vector<std::vector<float>>& latency_matrix, int device_count) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "                    Summary Statistics                    " << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::vector<float> valid_latencies;
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            if (i != j && latency_matrix[i][j] > 0) {
                valid_latencies.push_back(latency_matrix[i][j]);
            }
        }
    }
    
    if (valid_latencies.empty()) {
        std::cout << "No valid P2P connections found." << std::endl;
        return;
    }
    
    // Calculate statistics
    float min_latency = *std::min_element(valid_latencies.begin(), valid_latencies.end());
    float max_latency = *std::max_element(valid_latencies.begin(), valid_latencies.end());
    
    float sum = 0.0f;
    for (float lat : valid_latencies) {
        sum += lat;
    }
    float avg_latency = sum / valid_latencies.size();
    
    std::cout << "Total P2P connections: " << valid_latencies.size() << std::endl;
    std::cout << "Minimum latency: " << std::fixed << std::setprecision(3) << min_latency << " us" << std::endl;
    std::cout << "Maximum latency: " << std::fixed << std::setprecision(3) << max_latency << " us" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(3) << avg_latency << " us" << std::endl;
}

int main() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Found %d CUDA devices\n", deviceCount);
    
    if (deviceCount < 2) {
        printf("Need at least 2 CUDA devices for P2P testing\n");
        return 1;
    }
    
    // Initialize latency matrix
    std::vector<std::vector<float>> latency_matrix(deviceCount, std::vector<float>(deviceCount, -1.0f));
    
    // Enable P2P access for all device pairs
    int access;
    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            CUDA_CHECK(cudaSetDevice(i));
            cudaDeviceCanAccessPeer(&access, i, j);
            if (access) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
                printf("Enabled P2P access: Device %d -> Device %d\n", i, j);
            } else {
                printf("Device %d cannot access device %d\n", i, j);
            }
        }
    }
    
    // Test all device pairs and collect latency data
    printf("\nTesting P2P latency for all device pairs...\n");
    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            
            CUDA_CHECK(cudaSetDevice(i));
            cudaDeviceCanAccessPeer(&access, i, j);
            if (access) {
                printf("\n--- Testing P2P latency between device %d and %d ---\n", i, j);
                float latency = profile(i, j);
                latency_matrix[i][j] = latency;
                printf("Result: %.3f us\n", latency);
            }
        }
    }
    
    // Print results in table format
    print_latency_table(latency_matrix, deviceCount);
    print_summary_stats(latency_matrix, deviceCount);
    
    return 0;
}