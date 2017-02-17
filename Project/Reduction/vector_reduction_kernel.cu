/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#include <params.h>

__device__ unsigned int get_smid(void)
{
  unsigned int  ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}


__global__ void reduction_kernel(double * g_data, sm_queue_t* sm_queues, double * pre_write_back_data)
{
  const unsigned int smid = get_smid();
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  __shared__ double partialSum[SHARED_MEM_PER_BLOCK - 1]; // EXACT FIT OF 2 BLOCKS FOR BLOCK SIZE OF 256
  //__shared__ unsigned int shared_location;
  //__syncthreads();
  int my_loc = 0;
  if (sm_queues[smid].queue[my_loc] == 0)
  { //FIRST EVER BLOCK FOR THE SHARED MEM SECTION
    partialSum[t]              = g_data[start + t];
    partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];

    sm_queues[smid].queue[my_loc] = partialSum;

    for (unsigned int stride = blockDim.x; stride>=1024; stride >>= 1)
    {// THE REDUCTION
      __syncthreads();
      if (t < stride)
        partialSum[t] += partialSum[t+stride];
    }
  }
  else
  { // LATER BLOCKS
    if (t  < 1024)
    {
      partialSum[t]             += g_data[start + t];
      partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];
    }
    else
    {// BRING OTHER VALUES AS THEY ARE FROM GLOBAL TO SHARED
      partialSum[t]              = g_data[start + t];
      partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];
    }

    for (unsigned int stride = blockDim.x; stride>=1024; stride >>= 1)
    {// THE REDUCTION
      __syncthreads();
      if (t < stride)
        partialSum[t] += partialSum[t+stride];
    }
  }
}

__global__ void write_back_kernel(double* post_write_back_data)
{
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  __shared__ double partialSum[SHARED_MEM_PER_BLOCK -1];
  //__shared__ unsigned int shared_location;
  for (unsigned int stride = blockDim.x; stride>=1; stride >>= 1)
  {// THE REDUCTION
     __syncthreads();
     if (t < stride)
     partialSum[t] += partialSum[t+stride];
  }
  if (t==0)
    post_write_back_data[blockIdx.x] = partialSum[t];
}

__global__ void test_kernel(double * data, double* pre_write_back_data )
{
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ double partialSum[SHARED_MEM_PER_BLOCK];
  partialSum[t]                              = data[start + t];
  partialSum[blockDim.x + t]                 = data[start + blockDim.x + t];
  //  pre_write_back_data[i]                 = data[i];
  pre_write_back_data[start+t]               = partialSum[t];
  pre_write_back_data[start+t + blockDim.x ] = partialSum[t + blockDim.x];
}

__global__ void test_write_back_kernel(double* post_write_back_data)
{
  unsigned int t = threadIdx.x;
  __shared__ double partialSum[SHARED_MEM_PER_BLOCK];
  unsigned int start = 2*blockIdx.x*blockDim.x;
  //unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
  post_write_back_data[start + t]            = partialSum[t];
  post_write_back_data[start+t + blockDim.x ] = partialSum[t + blockDim.x];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
