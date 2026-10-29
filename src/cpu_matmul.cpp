#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace std::chrono;

void matmul_cpu(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void init_matrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main()
{
    const int N = 1024;
    const int WARMUP_RUNS = 1;
    const int BENCHMARK_RUNS = 5;

    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    srand(42);
    init_matrix(A, N);
    init_matrix(B, N);

    // ---------- Warm-up ----------
    for (int i = 0; i < WARMUP_RUNS; ++i)
    {
        matmul_cpu(A, B, C, N);
    }

    // ---------- Benchmark runs ----------
    vector<double> times_ms;

    for (int i = 0; i < BENCHMARK_RUNS; ++i)
    {
        auto start = high_resolution_clock::now();
        matmul_cpu(A, B, C, N);
        auto end = high_resolution_clock::now();

        duration<double, milli> elapsed = end - start;
        times_ms.push_back(elapsed.count());
    }

    // Calculate statistics
    double sum = 0.0;
    double min_time = times_ms[0];

    for (double t : times_ms)
    {
        sum += t;
        if (t < min_time)
            min_time = t;
    }

    double avg_time = sum / BENCHMARK_RUNS;
    double flops = 2.0 * N * N * N;
    double avg_gflops = (flops / (avg_time / 1000.0)) / 1e9;
    double best_gflops = (flops / (min_time / 1000.0)) / 1e9;

    // Print results
    cout << fixed << setprecision(3);
    cout << "=== CPU Single-Core Matrix Multiplication ===" << endl;
    cout << "Matrix size           : " << N << " x " << N << endl;
    cout << "Warm-up runs          : " << WARMUP_RUNS << endl;
    cout << "Benchmark runs        : " << BENCHMARK_RUNS << endl;
    cout << "Average time          : " << avg_time << " ms" << endl;
    cout << "Best time             : " << min_time << " ms" << endl;
    cout << "Average performance   : " << avg_gflops << " GFLOPS" << endl;
    cout << "Best performance      : " << best_gflops << " GFLOPS" << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}