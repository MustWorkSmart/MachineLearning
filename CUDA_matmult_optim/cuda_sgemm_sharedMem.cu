// sharedMem here as compared to naive sgemm and globalMemCoalescedWithinWarp earlier (*)
// let's see how much slower than cublas (cublas_sgemm_alpha1_beta0.cu) and how much faster than globalMemCoalescedWithinWarp.cu
// C = alpha*(matrix multiplication of A and B) + beta*C
// A is of size M by K, M rows, K columns
// B is of size K by N, K rows, N columns
// then C is M by N
// for now, C is NOT initialized as beta is set to 0.0 below; alpha set to 1.0

// (*)
//basically, the next step here is to use the smaller but faster (lower latency and higher bandwidth than global memory) shared memory (SMEM)
//idea is to have BLOCKSIZE x BLOCKSIZE of the matrices in such shared mem, and have work related to such done as much as possible, before loading up the next block
//e.g. to calculate the 1st 32x32 block of C, the 1st super-row (i.e. k/32 32x32 blocks) of A and the 1st super-column (i.e. n/32 32x32 blocks) of B are needed, but only 1 block from A and 1 block from B can be in the shared mem at a time due to its capacity (*)
// (*) note that parameters/codes here may NOT be tuned (or generalized) for such capacity and occupancy, like for various m/k/n/BLOCKSIZE (could have different dimensions in x/y) values and/or each element data type/size (fixed to float32 here)

/*
steps:

1) nvcc cuda_sgemm_sharedMem.cu -o cuda_sgemm_sharedMem

2) .\cuda_sgemm_sharedMem

...

3) ncu -o profile cuda_sgemm_sharedMem.exe

4) launch Nsight Compute, and drag profile.ncu-rep obtained from the last step to Nsight Compute
*/

//#include <cstdio>
//#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

// this is just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
   do { \
         cudaError_t __err = cudaGetLastError(); \
         if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
         } \
   } while (0)

void print_matrix(const int &m, const int &n, const float *A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[i * n + j]);
        }
        std::printf("\n");
    }
}

const int BLOCKSIZE = 32; // CUDA maximum is 1024 total threads per block, 2d block with 32 in each x/y dimension
//const int BLOCKSIZE_X = BLOCKSIZE; //not used for now
//const int BLOCKSIZE_Y = BLOCKSIZE; //not used for now
//const int BLOCKSIZE_K = BLOCKSIZE; //not used for now
//const int DSIZE = 2; //small inputs as initial test - tested and matching expected result (way below)
//now, trying a larger dataset
//const int DSIZE = 10; //tested and verified
//const int DSIZE = 10000;
//and, trying a much larger dataset
const int DSIZE = 1024*16;

/*
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
*/
/*
__global__ void sgemm_globalMem(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {
    // compute position in C that this thread is responsible for
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
*/
__global__ void sgemm_sharedMem(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {

    const int y = threadIdx.x;
    const int x = threadIdx.y;
    const int C_rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const int C_colIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // advance pointers to the starting positions
    A += blockIdx.y * BLOCKSIZE * K;
    B += blockIdx.x * BLOCKSIZE;
    C += blockIdx.y * BLOCKSIZE * N + blockIdx.x * BLOCKSIZE;

    // Cache a block (or tile) of A and B in shared memory for faster data reuse
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float tmp = 0.0;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Have each thread load one of the elements in A & B from global memory into shared memory.
        // Make y (=threadIdx.x) the consecutive index to allow global memory access coalescing
        As[x * BLOCKSIZE + y] = A[x * K + y];
        Bs[x * BLOCKSIZE + y] = B[x * N + y];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk (i.e. block of BLOCKSIZE * BLOCKSIZE)
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[x * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + y];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }

    if (C_rowIdx < M && C_colIdx < N)
    {
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main(int argc, char *argv[]) {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // these are just for timing
    clock_t t0, t1, t2;
    double t1sum=0.0;
    double t2sum=0.0;
    // start timing
    t0 = clock();

    const int m = DSIZE;
    const int n = DSIZE;
    const int k = DSIZE;

    h_A = new float[m*k]; // matrix A with size m * k
    h_B = new float[k*n];
    h_C = new float[m*n];

    float Aval, Bval;

//small inputs as initial test - tested and matching expected result (way below)
    /*
     *   A = | 1.0 | 3.0 |
     *       | 2.0 | 4.0 |
     *
     *   B = | 5.0 | 7.0 |
     *       | 6.0 | 8.0 |
     */
    //Aval = 1.0;
    //Bval = 5.0;
//small inputs as initial test - end

//more inputs to try next
    Aval = 1.0;
    Bval = 100.0;
//more inputs - end

    for (int x = 0; x < k; x++) {
        for (int y = 0; y < m; y++) {
            h_A[y * k + x] = Aval++;
        }
    }
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < k; y++) {
            h_B[y * n + x] = Bval++;
        }
    }

    std::vector<float> C(m * n);
    const float alpha = 1.0;
    const float beta = 0.0;

    printf("m is %d .. k is %d .. n is %d\n", m, k, n);

    if (m<=10 && k<=10 && n<=10) {
        printf("A\n");
        print_matrix(m, k, h_A);
        printf("=====\n");

        printf("B\n");
        print_matrix(k, n, h_B);
        printf("=====\n");    
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data over to GPU
    cudaMalloc(&d_A, m*k*sizeof(float));
    cudaMalloc(&d_B, k*n*sizeof(float));
    cudaMalloc(&d_C, m*n*sizeof(float));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_A, h_A, m*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    // compute - launch kernel
    // create as many blocks as necessary to map all of C
    //dim3 gridDim((m + block_size - 1)/block_size, (n + block_size - 1)/block_size, 1);
    dim3 gridDim((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE, 1);

    // 32 * 32 = 1024 thread per block
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    sgemm_sharedMem<<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */
    if (m<=10 && k<=10 && n<=10) {
        printf("C\n");
        print_matrix(m, n, h_C);
        printf("=====\n");
    }

    printf("First and Last elements of A are: %.2f and %.2f\n", h_A[0], h_A[m * k - 1]);
    printf("First and Last elements of B are: %.2f and %.2f\n", h_B[0], h_B[k * n - 1]);
    printf("First and Last elements of C are: %.2f and %.2f\n", h_C[0], h_C[m * n - 1]);

    /* free resources */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("===== The End =====\n");

    return EXIT_SUCCESS;
}