#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>

using namespace std;

#define TILE_SIZE 16

// -------------------------------------------------
// Tiled Matrix Multiplication using Shared Memory
// -------------------------------------------------
__global__ void matmul_tiled(const float *A, const float *B, float *C, int N)
{
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x; // Thread x index within the block
    int ty = threadIdx.y; // Thread y index within the block

    // Global row and column this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // Load one element of A tile into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < N && aCol < N)
            As[ty][tx] = A[row * N + aCol];
        else
            As[ty][tx] = 0.0f;

        // Load one element of B tile into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // Synchronize to make sure the tile is fully loaded
        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result
    if (row < N && col < N)
    {
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

    // Host memory
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    srand(42);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Block and Grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < WARMUP_RUNS; ++i)
    {
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    float total_ms = 0.0f;
    float min_ms = 1e9;

    for (int i = 0; i < BENCHMARK_RUNS; ++i)
    {
        CHECK_CUDA(cudaEventRecord(start));
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        total_ms += milliseconds;
        if (milliseconds < min_ms)
            min_ms = milliseconds;
    }

    float avg_ms = total_ms / BENCHMARK_RUNS;

    double flops = 2.0 * N * N * N;
    double avg_gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    double best_gflops = (flops / (min_ms / 1000.0)) / 1e9;

    // Print results
    cout << fixed << setprecision(3);
    cout << "=== CUDA Tiled Matrix Multiplication (Shared Memory) ===" << endl;
    cout << "Matrix size           : " << N << " x " << N << endl;
    cout << "Tile size             : " << TILE_SIZE << " x " << TILE_SIZE << endl;
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