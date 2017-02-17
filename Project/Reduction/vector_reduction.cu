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

#ifdef _WIN32
#  define NOMINMAX
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>
// includes, project
#include <cutil.h>
#include <params.h>
// includes, kernels

#include "./vector_reduction_kernel.cu"

#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0
#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)
double mytimer(void)
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
   return( ((double) (tp.tv_sec - start)) + (tp.tv_usec-startu)/1000000.0 );
}



////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

//int ReadFile(double*, char* file_name);
double computeOnDevice(double* h_data, int array_mem_size, double* h_sm_block_sums);

extern "C"
void computeGold( double* reference, double* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv)
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    const unsigned int array_mem_size = sizeof( double) * num_elements;

    // allocate host memory to store the input data
    double* h_data = (double*) malloc( array_mem_size);
    double * h_sm_block_sums = (double *) malloc (NUM_SM*sizeof(double));

    // * No arguments: Randomly generate input data and compare against the
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {
        case 1:  // One Argument
            //errorM = ReadFile(h_data, argv[1]);
            errorM = 0;
            if(errorM != 1)
            {
                printf("Error reading input file! Check the source for this print\n");
                exit(1);
            }
        break;

        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i)
            {
                //h_data[i] = floorf(1000*(rand()/(double)RAND_MAX));
                h_data[i] = i;//floorf(rand()*1000);
            }
        break;
    }
    // compute reference solution
    double reference = 0.0f;
    computeGold(&reference , h_data, num_elements);
    printf("GOLD VALUE %f\n", reference);

    // **===-------- Modify the body of this function -----------===**
    double result = computeOnDevice(h_data, num_elements, h_sm_block_sums);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range
    // that can be exactly represented
    double epsilon = 0.0f;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %f  host: %f\n", result, reference);
    // cleanup memory
    free( h_data);
}


/*int ReadFile(double* M, char* file_name)
{
	unsigned int elements_read = NUM_ELEMENTS;
	if (cutReadFilef(file_name, &M, &elements_read, true))
        return 1;
    else
        return 0;
}*/


void compute_occupancy(int grid_size)
{
  // NUM OF THREADS
  int max_threads_per_block = 1024;
  int max_threads_on_sm = 1536;
  int block_size = BLOCK_SIZE;
  int max_blocks_per_sm;
  if (block_size == max_threads_per_block){
    printf("NUM_BLOCKS PER SM is 1 \n");
    printf("[PDS] ALLOCATE FULL SHARED MEM TO THE BLOCK \n");
    return;
  }
  else
  {
    max_blocks_per_sm = floor(max_threads_on_sm/block_size);
    printf("constraint due to threads: %d \n", max_blocks_per_sm);
  }
  // SHARED MEMORY
  int shared_mem_size = 49152;
  int shared_mem_per_block = SHARED_MEM_PER_BLOCK;
  int sh_mem_constraint = floor(shared_mem_size/shared_mem_per_block);
  printf("shared_mem_size: %d \n", shared_mem_size);
  printf("sh_mem_perblock : %d \n", shared_mem_per_block);
  printf("constraint due to sh_mem: %d \n", sh_mem_constraint);
  if ((sh_mem_constraint) < max_blocks_per_sm)
    max_blocks_per_sm = sh_mem_constraint;
// REGISTERS /*TBD*/

}
/*// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: double* h_data is both the input and the output of this function.*/

double computeOnDevice(double* h_data, int num_elements, double* h_sm_block_sums)
{
    double *data;
    double * pre_write_back_data;
    double * pre_write_back_data_h;
    double * post_write_back_data;
    double * post_write_back_data_h;
    sm_queue_t* sm_queues;
    sm_queue_t* sm_queues_h;
    int num_blocks = ceil((double)num_elements/(2*BLOCK_SIZE));
    int padded_num_elements = num_blocks*2*BLOCK_SIZE;
    printf("padded_num_elements %d \n", padded_num_elements);
    dim3 dim_block (BLOCK_SIZE, 1, 1);
    dim3 dim_grid (num_blocks, 1, 1);
    dim3 dim_grid_write_back ((CONCURRENT_BLOCKS*NUM_SM), 1, 1);
    compute_occupancy(dim_grid.x);

    pre_write_back_data_h = (double *)malloc (num_elements * sizeof(double));
    post_write_back_data_h = (double *)malloc (num_elements * sizeof(double));
    memset (pre_write_back_data_h, (double)0.0, num_elements);
    memset (post_write_back_data_h, (double)0.0, num_elements);

    //{ // INITIALIZATIONS, MALLOCS AND MEMSETS
    cudaMalloc((void **)&data, padded_num_elements * sizeof(double));
    //cudaMalloc((void **)&data, num_elements * sizeof(double));
    //cudaMemset(data, (double)0.0, num_elements * sizeof(double));
    cudaMemset(data, (double)0.0, padded_num_elements * sizeof(double));
    cudaMemcpy(data, h_data, num_elements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **) & pre_write_back_data, num_elements * sizeof(double));
    cudaMemset(pre_write_back_data, (double)0.0, num_elements * sizeof(double));
    cudaMalloc((void **) & post_write_back_data, num_elements * sizeof(double));
    cudaMemset(post_write_back_data, (double)0.0, num_elements * sizeof(double));

      cudaMalloc((void **)&sm_queues,  NUM_SM * sizeof(sm_queue_t));
    //}
      sm_queues_h = (sm_queue_t *) malloc ( NUM_SM * sizeof(sm_queue_t));

      { // ALLOCATION OF MEMORY FOR EACH double** IN THE sm_queues
        for (int i = 0 ; i < NUM_SM; i++)
        {
          double** dummy;
          cudaMalloc((void**)&dummy, CONCURRENT_BLOCKS * sizeof(double*));
          sm_queues_h[i].queue = (double **) malloc (CONCURRENT_BLOCKS * sizeof(double*));
          cudaMemset(dummy, (int)0, CONCURRENT_BLOCKS*sizeof(double*));
          cudaMemcpy(&sm_queues[i].queue, &dummy, sizeof(double**), cudaMemcpyHostToDevice);
          cudaMemset(&sm_queues[i].front, 0, sizeof(unsigned int));
          cudaMemset(&sm_queues[i].back, 0, sizeof(unsigned int));
        }
      }
    /*//{// TEST Kernel Call
      test_kernel <<<dim_grid, dim_block>>> (data, pre_write_back_data);
      cudaDeviceSynchronize();
      test_write_back_kernel <<<dim_grid, dim_block>>> (post_write_back_data);
      cudaDeviceSynchronize();
    //}*/
    // KERNEL CALL
    unsigned int timer;
    double my_kernel_time = 0.0;
    double t0 = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    cutStartTimer(timer);
    TICK();
    reduction_kernel <<<dim_grid, dim_block>>> (data,  sm_queues, pre_write_back_data);
    write_back_kernel <<<dim_grid_write_back, 512>>> (post_write_back_data);
    cudaDeviceSynchronize();
    TOCK(my_kernel_time);
    cutStopTimer(timer);
    printf("my_kernel_time : %f \n", my_kernel_time);
    printf("CUDA Processing time : %f (ms) \n", cutGetTimerValue(timer) );


    //copy back
    cudaMemcpy(pre_write_back_data_h, pre_write_back_data , num_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(post_write_back_data_h, post_write_back_data , num_elements * sizeof(double), cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < NUM_SM ; i ++)
    {
      cudaMemcpy(&sm_queues_h[i], &sm_queues[i], sizeof(sm_queue_t),  + blockDim.xcudaMemcpyDeviceToHost);
    }*/

    /*for(int i =0; i < (dim_block.x) * (dim_grid.x) ; i++)
    {
      //printf("address: %p \n", block_array_h[i]);
      //printf("initial_data: %f , final_data: %f\n",h_data[i], data_h[i]);
      printf("pre_write_back_data : %f, post_write_back_data : %f \n", pre_write_back_data_h[i], post_write_back_data_h[i]);
      //printf("pre_write_back_data : %f\n", pre_write_back_data_h[i]);
    }*/

    double sum = 0;
    for(int i = 0; i < dim_grid_write_back.x; i++)
    {
      sum += post_write_back_data_h[i];
    }

   printf("Num_of_blocks %d \n", dim_grid.x);
   printf("Num Elements %d \n", num_elements);

  return sum;
}

