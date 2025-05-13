// smem_2DThreadTiling here improved on top of smem_1DThreadTiling earlier
// further reducing the number of GMEM/SMEM accesses by calculating even more results per thread
// .. which is optimizing on arithmetic intensity (AI)
// so, each thread now works on [TMxTN=8x8] C elements (instead of TM[=8]x1 elements in last kernel) at a time
// note: BM/BN dimensions changed (TN added too), and will need further perf experiments on various sizes combos
// before: each threadblock is 512x1 and smem is for 64x8 block/tile of A and 8x64 block/tile of B
// now: each threadblock is 256x1 and smem is for 32x8 block/tile of A and 8x32 block/tile of B
// with BM=BN=128, each threadblock is for 128x128 elements, while each thread is processing TMxTN=8x8 elements
// .. hence having 128x128 / 8x8 = 256 threads in each thread block, configured all in x dimension

// C = alpha*(matrix multiplication of A and B) + beta*C
// A is of size M by K, M rows, K columns
// B is of size K by N, K rows, N columns
// then C is M by N
// for now, C is NOT initialized as beta is set to 0.0 below; alpha set to 1.0

/*
steps:

1) nvcc cuda_sgemm_smem_2DThreadTiling.cu -o cuda_sgemm_smem_2DThreadTiling

2) .\cuda_sgemm_smem_2DThreadTiling

3) ncu -o profile cuda_sgemm_smem_2DThreadTiling.exe

4) launch Nsight Compute, and drag profile.ncu-rep obtained from the last step to Nsight Compute
*/

//#include <cstdio>
//#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
//#include <cstdint>
#include <stdint.h>

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

//const int BLOCKSIZE = 32; // CUDA maximum is 1024 total threads per block, 2d block with 32 in each x/y dimension
//const int BLOCKSIZE_X = BLOCKSIZE; //not used for now
//const int BLOCKSIZE_Y = BLOCKSIZE; //not used for now
//const int BLOCKSIZE_K = BLOCKSIZE; //not used for now
//const int BM = 64;
//const int BN = 64;
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

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
__global__ void sgemm_smem_2DThreadTiling(int M, int N, int K, float alpha, const float* A,
    const float* B, float beta, float* C) {

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    //const int threadCol = threadIdx.x % BN;
    //const int threadRow = threadIdx.x / BN;
    // BN/TN are the number of threads to span a column - 128/8 = 16
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int strideA = (BM * BN) / (TM * TN) / BK; // 32
    const int strideB = (BM * BN) / (TM * TN) / BN; // 2
    int C_rowIdx;
    int C_colIdx;

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int innerColA = threadIdx.x % BK; // 8 columns since BK==8
    const int innerRowA = threadIdx.x / BK; // .. 256/8 = 32 rows -> 32x8 A blocks (*) loaded per threadblock
    const int innerColB = threadIdx.x % BN; // 128 columns since BN==128
    const int innerRowB = threadIdx.x / BN; // .. 256/128 = 2 rows -> 2x128 B blocks (*) loaded per threadblock
    // (*) blocks, NOT block, due to the new loops (see the ones with loadOffset and stride*) below

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    // outer loop over block tiles
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        //As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA]; //last 1DThread Tiling kernel
        //Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB]; //last 1DThread Tiling kernel
        // .. with loadOffset and strideA=32, now each thread-block loading:
        // 4 32x8 A blocks -> 128x8 block of A (instead of just 1 32x8 A block)
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        // .. with loadOffset and strideB=2, now each thread-block loading: 
        // 4 2x128 B blocks -> 8x128 block of B (instead of just 1 2x128 B block)
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        // advance blocktile for next iteration in the outer loop above (A/B not used rest of outer loop)
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        /* //last 1DThread Tiling kernel
        // we make the dotproduct loop the outside loop, which facilitates
        // reuse of the Bs entry, which we can cache in a tmp var.
            float tmp = Bs[dotIdx * BN + threadCol];
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmp;
            }
        */
            // load relevant As & Bs entries into registers
            // -> a column of size TM from As, a row of size TN from Bs
            for (int i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            // perform outer product on register cache, accumulate into threadResults thru the outer loop
            for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    /*
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        C_rowIdx = blockIdx.y * BM + threadIdx.x / BN * TM + resIdx;
        C_colIdx = blockIdx.x * BN + threadIdx.x % BN;
        if (C_rowIdx < M && C_colIdx < N) {
            C[(threadRow * TM + resIdx) * N + threadCol] =
                alpha * threadResults[resIdx] +
                beta * C[(threadRow * TM + resIdx) * N + threadCol];
        }
    }
    */
    for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C_rowIdx = blockIdx.y * BM + threadRow * TM + resIdxM;
            C_colIdx = blockIdx.x * BN + threadCol * TN + resIdxN;
            if (C_rowIdx < M && C_colIdx < N) {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                    alpha * threadResults[resIdxM * TN + resIdxN] +
                    beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
            }
        }
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
    //dim3 gridDim((n + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    dim3 gridDim((n + BN - 1) / BN, (m + BM - 1) / BM, 1);

    // 32 * 32 = 1024 threads per block
    //dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
    // 64 * 64 / 8 = 512 threads per block
    dim3 blockDim(BM*BN/(TM*TN), 1, 1);

    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    sgemm_smem_2DThreadTiling<<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);

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