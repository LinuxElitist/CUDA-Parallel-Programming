#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//#include <cutil.h>
#include "util.h"
#include "cpu_hist.h"
#include "static_hist.h"

#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0
float mytimer(void)
{
   struct timeval tp;
   static long start=0, startu;
   if (!start)
   {
      gettimeofday(&tp, NULL);
      start = tp.tv_sec;
      startu = tp.tv_usec;
      return(0.0);
   }
   gettimeofday(&tp, NULL);
   return( ((float) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
}

__global__ void count_bins(uint8_t *d_input, uint32_t *histo_bins);
__global__ void merge_bins(uint32_t * histo_bins);
__global__ void init_bins();

dim3 grid_dim(ceil((INPUT_HEIGHT * INPUT_WIDTH) / BLOCK_SIZE),1,1);
dim3 block_dim(BLOCK_SIZE,1,1);
dim3 write_back_grid (CONCURRENT_BLOCKS*NUM_SM,1,1);
//dim3 write_back_grid (15,1,1);

uint8_t * d_input;
uint32_t * histo_bins;
float kernel_time = 0.0;

inline __device__ void addByte(uint *s_WarpHist, uint data){
    atomicAdd(s_WarpHist + data, 1);
}

// s_WarpHist is the pointer to the histogram for the current warp
// data is of uint type and contains 4 data values (32 = 8*4)
inline __device__ void addWord(uint *s_WarpHist, uint data){
    addByte(s_WarpHist, (data >>  0) & 0xFFU);
    addByte(s_WarpHist, (data >>  8) & 0xFFU);
    addByte(s_WarpHist, (data >> 16) & 0xFFU);
    addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__device__ unsigned int get_smid(void)
{
  unsigned int  ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}

void AllocateOnDevice(uint8_t ** h_input)
{
  cudaMalloc((void **)&d_input, IN_BYTE_COUNT);
  for(int i = 0; i < INPUT_HEIGHT; ++i)
    cudaMemcpy((d_input + i*INPUT_WIDTH), h_input[i], INPUT_WIDTH * sizeof(uint8_t), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&histo_bins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
	cudaMemset(histo_bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
	cudaDeviceSynchronize();
}

void kernel_call()
{
  cudaMemset(histo_bins, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));
  float t0 = 0.0;
  printf("Input Size:       %d\n",INPUT_WIDTH*INPUT_HEIGHT);
  printf("Histo Size:       %d\n",HISTO_WIDTH*HISTO_HEIGHT);
  printf("Max Concurrent Blocks per SM:       %d\n",CONCURRENT_BLOCKS);
  printf("compute_grid_dim: %d\n", grid_dim.x);
  printf("write_grid_dim:   %d\n", write_back_grid.x);
  TICK();
  init_bins<<<write_back_grid, block_dim>>>();
  count_bins<<<grid_dim, block_dim>>> (d_input, histo_bins);
  merge_bins<<<write_back_grid, block_dim>>> (histo_bins);
  cudaDeviceSynchronize();
  TOCK(kernel_time);
}

void static_hist()
{
  TIME_IT ("static_hist", NUM_ITERS, kernel_call();)
}

__global__ void init_bins()
{
 __shared__ uint32_t private_histogram[HISTO_SIZE];
  int i = threadIdx.x;
  while (i < (HISTO_HEIGHT * HISTO_WIDTH))
  {
    private_histogram[i] = 0;
    i += blockDim.x;
  }
}

/* Generated input is 8bit vals 
   Each thread loads 4 8bit vals as
   32bit val and shifts bits */
__global__ void count_bins(uint8_t *d_input, uint32_t *histo_bins)
{
  int t = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int smid = get_smid();
  __shared__ uint32_t  private_histogram[HISTO_SIZE];
  if (t < IN_CHUNK_COUNT)
  {
    uint32_t data = d_input[t];
    addWord(private_histogram, data);
    //atomicAdd(&private_histogram[(d_input[t])], 1);
  }
}

__global__ void merge_bins(uint32_t * histo_bins)
{

  __shared__ uint32_t private_histogram[HISTO_SIZE];
  const unsigned int smid = get_smid();
  int i = threadIdx.x;
  while (i < (HISTO_HEIGHT * HISTO_WIDTH))
  {
    atomicAdd(&histo_bins[i], private_histogram[i]);
    i += blockDim.x;
  }
}

void CopyFromDevice(uint32_t *kernel_bins)
{
  cudaDeviceSynchronize();
  cudaMemcpy(kernel_bins, histo_bins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

void FreeOnDevice()
{
    cudaFree(d_input);
    cudaFree(histo_bins);
}
