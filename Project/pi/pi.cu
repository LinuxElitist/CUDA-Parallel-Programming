#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h> 
using namespace std;

/*Error handling Macro for various GPU functions*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void pi(curandState *,  unsigned long long int *, int ,  unsigned long long int *);
void pi_reset( unsigned long long int *, unsigned long long int *,curandState *);

/*setupKernel for the random number initilization using 
  curand libraries assigning random number to each thread  */

__global__ void setupKernel(curandState *state)

{
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	curand_init(123456789,index,0,&state[index]);

}

/*MonteCarloEstimation Kernel for random number generation of random numbers 
  and reduction to get the value of pi from given point */

__global__ void MonteCarloKernel(curandState *state,  unsigned long long int *count,int m )
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int cache[256];
	cache[threadIdx.x]=0;

	__syncthreads();
	unsigned int tmp=0;
	while(tmp < m)
	{
		double x=curand_uniform_double(&state[index]);
		double y=curand_uniform_double(&state[index]);
		double r= x*x + y*y;

		if(r<=1)
		{
			cache[threadIdx.x]++;
		}
		tmp++;
	}

	__syncthreads();

	/*Sum Reduction code generated for count  */
	unsigned int i =blockDim.x/2;
	while(i!=0)
	{
		if (threadIdx.x<i)
		{
			cache[threadIdx.x]+=cache[threadIdx.x+i];
		}
		i = i/2; 
		__syncthreads();
	}

	//global variable update using atomic operations 
	if(threadIdx.x==0){
		atomicAdd(count,cache[0]);
	}
}

void pi_init( )

{

	unsigned long long int *h_count;
	unsigned long long int *d_count;
	curandState *d_state;
	int n=256*256;
	int m =32768;
	h_count = ( unsigned long long int *) malloc(sizeof( unsigned long long int));
	gpuErrchk(cudaMalloc ((void**) &d_count,sizeof( unsigned long long int)));
	gpuErrchk(cudaMalloc ((void**) &d_state,n*sizeof(curandState)));
	gpuErrchk(cudaMemset(d_count,0,sizeof( unsigned long long int)));


	dim3 gridsizee =256;
	dim3 blocksize=256;
	setupKernel<<<gridsizee,blocksize>>>(d_state);
	gpuErrchk(cudaDeviceSynchronize()); 

	pi(d_state,h_count,m,d_count);

	double pival;
	pival= (*h_count*4.0)/((n+0.0)*(m+0.0));
	cout.precision(17);
	/*Display results for gpu*/
	cout<<"GPU approximated Pi value is:"<<fixed<<pival<<endl;
	pi_reset(h_count,d_count,d_state);
}
void pi(curandState *d_state,  unsigned long long int *h_count, int m,  unsigned long long int *d_count)
{	
	dim3 gridsizee =256;
	dim3 blocksize=256;

	/*Executing monte carlo kernel*/
	MonteCarloKernel<<<gridsizee, blocksize>>>(d_state, d_count, m);
	gpuErrchk(cudaMemcpy(h_count, d_count, sizeof( unsigned long long int), cudaMemcpyDeviceToHost));

}

void pi_reset( unsigned long long int *h_count,  unsigned long long int * d_count, curandState *d_state)
{

	free(h_count);
	cudaFree(d_count);
	cudaFree(d_state);
}

int main()
{

	pi_init();
	return 1;
}

