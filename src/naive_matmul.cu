#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>

using namespace std;

// -------------------------------------------------
// Naive CUDA Kernel
// Each thread computes one element of C
// -------------------------------------------------
__global__ void matmul_naive(const float *A, const float *B, float *C, int N)
{
    // Calculate the row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// -------------------------------------------------
// Initialize matrix with random values
// -------------------------------------------------
void init_matrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// -------------------------------------------------
// CUDA error checking macro
// -------------------------------------------------
#define CHECK_CUDA(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

int main()
{
    const int N = 1024;
    const int WARMUP_RUNS = 2;
    const int BENCHMARK_RUNS = 5;
    const size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    srand(42);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data from Host to Device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions (16x16 threads per block)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for accurate GPU timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ---------- Warm-up runs ----------
    for (int i = 0; i < WARMUP_RUNS; ++i)
    {
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---------- Benchmark runs ----------
    float total_ms = 0.0f;
    float min_ms = 1e9;

    for (int i = 0; i < BENCHMARK_RUNS; ++i)
    {
        CHECK_CUDA(cudaEventRecord(start));
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        total_ms += milliseconds;
        if (milliseconds < min_ms)
            min_ms = milliseconds;
    }

    float avg_ms = total_ms / BENCHMARK_RUNS;

    // Calculate GFLOPS
    double flops = 2.0 * N * N * N;
    double avg_gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    double best_gflops = (flops / (min_ms / 1000.0)) / 1e9;

    // Print results
    cout << fixed << setprecision(3);
    cout << "=== CUDA Naive Matrix Multiplication ===" << endl;
    cout << "Matrix size           : " << N << " x " << N << endl;
    cout << "Block size            : " << blockDim.x << " x " << blockDim.y << endl;
    cout << "Grid size             : " << gridDim.x << " x " << gridDim.y << endl;
    cout << "Warm-up runs          : " << WARMUP_RUNS << endl;
    cout << "Benchmark runs        : " << BENCHMARK_RUNS << endl;
    cout << "Average kernel time   : " << avg_ms << " ms" << endl;
    cout << "Best kernel time      : " << min_ms << " ms" << endl;
    cout << "Average performance   : " << avg_gflops << " GFLOPS" << endl;
    cout << "Best performance      : " << best_gflops << " GFLOPS" << endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}