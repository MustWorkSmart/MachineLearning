//this is placeholder for future to-do
//faster#7 and more – Warp Tiling, Tensor Core, CUTLASS (https://github.com/NVIDIA/cutlass)

// cuda_sgemm_warpTiling.cu - to improve on top of cuda_sgemm_vectorizedMemFloat4_2DThreadTiling.cu by:
// Warp Tiling
// FYI:
// warpId=threadIdx.x % warpSize, where warpSize is a built-in variable that is equal to 32
// Warps are the unit of scheduling that is mapped to the warp-schedulers that are part of the SM
// Shared-memory bank conflicts happen only between threads that are in the same warp