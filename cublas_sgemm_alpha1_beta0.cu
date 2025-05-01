//based on (with many changes though):
//https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/gemm/cublas_gemm_example.cu
// C = alpha*(matrix multiplication of A and B) + beta*C
// for now, C is NOT initialized as beta is set to 0.0 below

/*
steps:

1) nvcc cublas_sgemm_alpha1_beta0.cu -o cublas_sgemm -lcublas

2) .\cublas_sgemm

m is 16384 .. k is 16384 .. n is 16384
First and Last elements of A are: 1.00 and 16777216.00
First and Last elements of B are: 100.00 and 16777216.00
First and Last elements of C are: 2275269125603328.00 and 4467711567839887360.00
===== The End =====

3) ncu -o profile cublas_sgemm.exe
*/

/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

//using data_type = double;
using data_type = float;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

//original inputs - tested and matching expected result (way below)
/*
    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
*/

    /* //they had this wrong online, fixed here
     *   A = | 1.0 | 3.0 |
     *       | 2.0 | 4.0 |
     *
     *   B = | 5.0 | 7.0 |
     *       | 6.0 | 8.0 |
     */
    // layout in cuBLAS : column-major order
   // const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
   // const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};

//new inputs
//now, trying a larger dataset
    //const int ds = 10; // ds stands for datasize //tested and verified
//and, trying a much larger dataset
    //const int ds = 10000;
    const int ds = 1024*16;

    const int m = ds;
    const int n = ds;
    const int k = ds;
    const int lda = ds;
    const int ldb = ds;
    const int ldc = ds;

    data_type Aval = static_cast<data_type>(1.0);
    data_type Bval = static_cast<data_type>(100.0);

    std::vector<data_type> A(m * k); // Initialize A with size m * k
    std::vector<data_type> B(k * n); // Initialize B with size k * n
    for (int i = 0; i < m * k; i++) {
        A[i] = (data_type) Aval++;
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = (data_type) Bval++;
    }

    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("m is %d .. k is %d .. n is %d\n", m, k, n);

    if (m<=10 && k<=10 && n<=10) {
        printf("A\n");
        print_matrix(m, k, A.data(), lda);
        printf("=====\n");

        printf("B\n");
        print_matrix(k, n, B.data(), ldb);
        printf("=====\n");    
    }

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(
        //cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
    //will need to make this dependable on data_type above

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */
    if (m<=10 && k<=10 && n<=10) {
        printf("C\n");
        print_matrix(m, n, C.data(), ldc);
        printf("=====\n");
    }

    printf("First and Last elements of A are: %.2f and %.2f\n", A[0], A[m * k - 1]);
    printf("First and Last elements of B are: %.2f and %.2f\n", B[0], B[k * n - 1]);
    printf("First and Last elements of C are: %.2f and %.2f\n", C[0], C[m * n - 1]);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    printf("===== The End =====\n");

    return EXIT_SUCCESS;
}
