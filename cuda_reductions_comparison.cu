/*
steps:

1) nvcc cuda_reductions_comparison.cu -o reductions

2) .\reductions

atomic sum reduction correct!
reduction shared mem parallel sweep sum correct!
reduction warp shuffle sum correct!
reduction 2-stage shared memory sum correct!

3)
ncu -o profile reductions

4)
open/launch Nsight Compute and drag profile.ncu-rep from #3 above to it
*/

/*
note that these "bad" approaches are NOT done below:
(1)
Does NOT work (as every thread is trying to do this, potentially at the same time):
*c += a[i];

(2)
Works but NOT parallel:
atomicAdd(&c, a[i]); https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
*/

#include <stdio.h>

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

//const size_t N = 8ULL*1024ULL*1024ULL;  // data size
//const size_t N = 256*640; // data size
//const size_t N = 163840; // smaller data size to avoid the grid stride loop taking most of the time
//but should still be large enough to allow enough threads saturating the machine (i.e. high occupancy)
//this is now set to the max# threads for V100 - 80 SMs * 2048 max threads per SM
const size_t N = 24*2048; //RTX 4060 has 24 SMs, 24*2048

const int BLOCK_SIZE = 256;  // CUDA maximum is 1024
// naive atomic reduction kernel
__global__ void atomic_red(const float *gdata, float *out){
   size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
   if (idx < N) atomicAdd(out, gdata[idx]);
}

__global__ void reduce_a(float *gdata, float *out){
      __shared__ float sdata[BLOCK_SIZE];
      int tid = threadIdx.x;
      sdata[tid] = 0.0f;
      size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

      while (idx < N) {  // grid stride loop to load data
         sdata[tid] += gdata[idx];
         idx += gridDim.x*blockDim.x;  
      }

      for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
         __syncthreads();
         if (tid < s)  // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
         }
      if (tid == 0) atomicAdd(out, sdata[0]);
   }

__global__ void reduce_ws(float *gdata, float *out){
   __shared__ float sdata[32];
   int tid = threadIdx.x;
   int idx = threadIdx.x+blockDim.x*blockIdx.x;
   float val = 0.0f;
   unsigned mask = 0xFFFFFFFFU;
   int lane = threadIdx.x % warpSize;
   int warpID = threadIdx.x / warpSize;
   while (idx < N) {  // grid stride loop to load 
      val += gdata[idx];
      idx += gridDim.x*blockDim.x;  
   }

   // 1st warp-shuffle reduction
   for (int offset = warpSize/2; offset > 0; offset >>= 1) 
      val += __shfl_down_sync(mask, val, offset);
   if (lane == 0) sdata[warpID] = val;
   __syncthreads(); // put warp results in shared mem

   // hereafter, just warp 0
   if (warpID == 0){
   // reload val from shared mem if warp existed
      val = (tid < blockDim.x/warpSize)?sdata[lane]:0;

 // final warp-shuffle reduction
      for (int offset = warpSize/2; offset > 0; offset >>= 1) 
         val += __shfl_down_sync(mask, val, offset);

      if  (tid == 0) atomicAdd(out, val);
   }
}

__global__ void reduce(float *gdata, float *out, size_t n){
   __shared__ float sdata[BLOCK_SIZE];
   int tid = threadIdx.x;
   sdata[tid] = 0.0f;
   size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

   while (idx < n) {  // grid stride loop to load data
      sdata[tid] += gdata[idx];
      idx += gridDim.x*blockDim.x;  
   }

   for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
      __syncthreads();
      if (tid < s)  // parallel sweep reduction
         sdata[tid] += sdata[tid + s];
   }
   if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main(){

  float *h_A, *h_sum, *d_A, *d_sum, *d_sums;
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  for (int i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  //const int blocks = 640; //N is 163840 and BLOCK_SIZE is 256 above, 640 is N/BLOCK_SIZE
  const int blocks = N / BLOCK_SIZE;
  cudaMalloc(&d_A, N*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sum, sizeof(float));  // allocate device space for sum
  cudaMalloc(&d_sums, blocks*sizeof(float));  // allocate device space for partial sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); return -1;}
  printf("atomic sum reduction correct!\n");

  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction shared mem parallel sweep kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction shared mem parallel sweep sum incorrect!\n"); return -1;}
  printf("reduction w/atomic sum correct!\n");

  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); return -1;}
  printf("reduction warp shuffle sum correct!\n");

  cudaMemset(d_sums, 0, blocks*sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N); // reduce stage 1
  cudaCheckErrors("reduction kernel launch failure");
  reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks); // reduce stage 2
  cudaCheckErrors("reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction 2-stage shared memory kernel execution failure or cudaMemcpy D2H failure");
  if (*h_sum != (float)N) {printf("reduction 2-stage shared memory sum incorrect!\n"); return -1;}
  printf("reduction 2-stage shared memory sum correct!\n");

  return 0;
}
  
