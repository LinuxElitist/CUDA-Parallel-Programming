/*This code implements Linear Regression */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//#include <cutil.h>

// include time stamp
#include <sys/time.h>
struct timeval begin, end; 
unsigned long long time_spent;

//global variables
#define NUM_ELEMENTS 262144 //must be k*BLOCK_SIZE
#define NUM_BLOCK 32
#define BLOCK_SIZE 1024 //this will make sure everytime only one block in SM

//normal linear_regression
__global__ void linear_regression(float *data_x, float *data_y, double *result, int n){
	//share memory size: 32KB, 1K*8*4 
        __shared__ double sum_xy[BLOCK_SIZE];
        __shared__ double sum_xx[BLOCK_SIZE];
        __shared__ double sum_x[BLOCK_SIZE];
	__shared__ double sum_y[BLOCK_SIZE];

        unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned tid = threadIdx.x;
        unsigned gridsize = blockDim.x * gridDim.x;

        double local_xy = 0;
        double local_xx = 0;
        double local_x = 0;
        double local_y = 0;

        while(id < n){ //need #threads is k*blocksize
		local_xy += data_x[id]*data_y[id]; 
		local_xx += data_x[id]*data_x[id];
		local_x += data_x[id]; 
		local_y += data_y[id]; 
                id += gridsize;
        }
        sum_xy[tid] = local_xy;
        sum_xx[tid] = local_xx;
        sum_x[tid] = local_x;
        sum_y[tid] = local_y;
        __syncthreads();

        for (unsigned stride = (blockDim.x/2); stride >= 1;  stride >>= 1) {
                if (tid < stride){ //need #threads is k*blocksize
                        sum_xy[tid] += sum_xy[tid + stride];
                        sum_xx[tid] += sum_xx[tid + stride];
                        sum_x[tid] += sum_x[tid + stride];
                        sum_y[tid] += sum_y[tid + stride];
                }
                __syncthreads();
        }

        if(tid == 0){
                result[blockIdx.x*4] = sum_xy[0];
                result[blockIdx.x*4+1] = sum_xx[0];
                result[blockIdx.x*4+2] = sum_x[0];
                result[blockIdx.x*4+3] = sum_y[0];
        }
}

/* CPU code */
int main(int argc, char **argv){
	float ans_b1, ans_b0; 
	if(argc != 3){
		printf("error argument input, should be: $xx_regression b0 b1, 2 float/int number \n");
		return 0; 
	}
	ans_b0 = atof(argv[1]);
	ans_b1 = atof(argv[2]);

        int i;
        float *h_data_x, *h_data_y; //host data x y
	//they are: sum(xi * yi), sum(xi^2), sum(xi), sum(yi)
	double h_A, h_B, h_C, h_D; 
	//they are: mean(x), mean(y)
	double temp_A, temp_B; 
	double h_b1, h_b0, d_b1, d_b0; 


        srand(time(NULL));
	h_A = 0; h_B = 0; h_C = 0; h_D = 0; 

        h_data_x = (float *)malloc(NUM_ELEMENTS * sizeof(float));
	h_data_y = (float *)malloc(NUM_ELEMENTS * sizeof(float));
//////
printf("\nHost data: \n");
        for(i = 0; i < NUM_ELEMENTS; i++){
                h_data_x[i] = (rand()%1000)/10.0; //x from 0-100
                h_data_y[i] = ans_b0 + ans_b1 * h_data_x[i] + (rand()%80 - 40)/10.0; //y will have +-4 change
		
		h_A += h_data_x[i]*h_data_y[i]; 
		h_B += h_data_x[i]*h_data_x[i]; 
		h_C += h_data_x[i]; 
		h_D += h_data_y[i]; 
//////
//printf("[%d,%d] ", i, h_data[i]); 
        }
//////
printf("\n");

	temp_A = h_C/NUM_ELEMENTS; //mean x
	temp_B = h_D/NUM_ELEMENTS; //mean y
	h_b1 = (h_A - NUM_ELEMENTS*temp_A*temp_B)/(h_B - NUM_ELEMENTS*temp_A*temp_A);
	h_b0 = temp_B - h_b1 * temp_A; 


	double result[NUM_BLOCK*4]; //for device h_A - h_D
	for(i = 0; i<(NUM_BLOCK*4); i++){
		result[i] = 0;
	}

gettimeofday(&begin, NULL);

        float *d_data_x, *d_data_y; //device data
        double *d_result;
        cudaMalloc((void**)&d_data_x, NUM_ELEMENTS*sizeof(float));
        cudaMalloc((void**)&d_data_y, NUM_ELEMENTS*sizeof(float));
        cudaMemcpy(d_data_x, h_data_x, NUM_ELEMENTS*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_data_y, h_data_y, NUM_ELEMENTS*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_result, NUM_BLOCK*4*sizeof(double));
        cudaMemcpy(d_result, result, NUM_BLOCK*4*sizeof(double), cudaMemcpyHostToDevice);
        dim3 dim_grid(NUM_BLOCK, 1, 1);
        dim3 dim_block(BLOCK_SIZE, 1, 1);

        linear_regression<<<dim_grid, dim_block>>>(d_data_x, d_data_y, d_result, NUM_ELEMENTS);
        cudaDeviceSynchronize();

        cudaMemcpy(result, d_result, NUM_BLOCK*4*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_data_x);
        cudaFree(d_data_y);
        cudaFree(d_result);

gettimeofday(&end, NULL);
time_spent = 1000000 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) ; 
printf("linear_regression (normal) computation cost time: %llu us \n", time_spent);

	h_A = 0; h_B = 0; h_C = 0; h_D = 0; 
        for(i = 0; i< NUM_BLOCK; i++){
		h_A += result[i*4]; 
		h_B += result[i*4 + 1]; 
		h_C += result[i*4 + 2]; 
		h_D += result[i*4 + 3]; 
        }
	temp_A = h_C/NUM_ELEMENTS; //mean x
	temp_B = h_D/NUM_ELEMENTS; //mean y
	d_b1 = (h_A - NUM_ELEMENTS*temp_A*temp_B)/(h_B - NUM_ELEMENTS*temp_A*temp_A);
	d_b0 = temp_B - d_b1 * temp_A; 

        printf("input: b0: %f, b1: %f \nhost: b0: %f, b1: %f \ndevice: b0: %f, b1: %f \n", 
		ans_b0, ans_b1, h_b0, h_b1, d_b0, d_b1);

        free(h_data_x);
        free(h_data_y);
        return 0;
}


