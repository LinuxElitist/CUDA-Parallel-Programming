/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <helper_cuda.h>
#include <math.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "bodysystem.h"

__constant__ float softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared,
                              &softeningSq,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

template<>
__device__ double rsqrt_T<double>(double x)
{
    return rsqrt(x);
}


// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i+blockDim.x*threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i,j) sharedPos[i+blockDim.x*j]

template <typename T>
__device__ T getSofteningSquared()
{
    return softeningSquared;
}
template <>
__device__ double getSofteningSquared<double>()
{
    return softeningSquared_fp64;
}

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};


/* SSM design include and functions START */
#include "cutil_ssm_config.h"

__global__ void SSM_CLEAN(){
        __shared__ float ssm[SSM_CLEAN_BLOCK_SIZE*SSM_CLEAN_EXPAND];
        //__shared__ double ssm[SSM_CLEAN_BLOCK_SIZE*SSM_CLEAN_EXPAND];
        int id = threadIdx.x;

        int i;
        for(i = 0; i<SSM_CLEAN_EXPAND; i++){
                ssm[id + SSM_CLEAN_BLOCK_SIZE*i] = 0.0;
        }

		//if (threadIdx.x == 0) printf("working on clean SSM initial! \n");
}

//this code doesn't need SSM_COLLECT(argument)
//but, why do not put the data into constant memory? 

/* SSM design include and functions END */




template <typename T>
__device__ typename vec3<T>::Type
bodyBodyInteraction(typename vec3<T>::Type ai,
                    typename vec4<T>::Type bi,
                    typename vec4<T>::Type bj)
{
    typename vec3<T>::Type r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

template <typename T>
__device__ typename vec3<T>::Type
computeBodyAccel(typename vec4<T>::Type bodyPos,
                 typename vec4<T>::Type *positions,
                 int numTiles)
{
	__shared__ typename vec4<T>::Type SSM[SSM_CLEAN_BLOCK_SIZE*SSM_CLEAN_EXPAND/8];
	//size if not 4*float !!
    //typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();
	typename vec4<T>::Type *sharedPos = SSM;

    typename vec3<T>::Type acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
	//////
	//load from global memory, for position
//////
//SSM design
	//if (threadIdx.x == 0) printf("working on calculation! block id: %d \n", blockIdx.x);

	//original
	//sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

	if(sharedPos[numTiles * blockDim.x].w == 0){
        	sharedPos[tile * blockDim.x + threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

			//if (threadIdx.x == 0) printf("loading data! block id: %d, %d, %f, %f, %f, %f \n", blockIdx.x, numTiles, 
			//	sharedPos[tile * blockDim.x + threadIdx.x].x, sharedPos[tile * blockDim.x + threadIdx.x].y, sharedPos[tile * blockDim.x + threadIdx.x].z, 
			//	sharedPos[tile * blockDim.x + threadIdx.x].w);

	}
	else{
		if(threadIdx.x == 0) printf("already load data! SSM reuse! \n");
	}
	//if (threadIdx.x == 0) printf("finish loading! block id: %d \n", blockIdx.x);

        __syncthreads();

	//if (threadIdx.x == 0) printf("after sync! block id: %d, %d \n", blockIdx.x, tile);

        // This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

	//////
	//calculate to current block data (like: matrix mul: tile to itself)
	//also calculate the entire "acc"
        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
//////
//SSM design
			//if (threadIdx.x == 0) printf("before calculation! block id: %d, sharedPos %d \n", blockIdx.x, (tile * blockDim.x + counter));

            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[tile * blockDim.x + counter]);
			//acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

	//if (threadIdx.x == 0) printf("finish calculation! block id: %d \n", blockIdx.x);

        __syncthreads();
    }

//////
//SSM design, set flag
    if(sharedPos[numTiles * blockDim.x].w == 0){
	  if(threadIdx.x == 0){
	    sharedPos[numTiles * blockDim.x].w = 1; 
	  }
    }

    return acc;
}

template<typename T>
__global__ void
integrateBodies(typename vec4<T>::Type *__restrict__ newPos,
                typename vec4<T>::Type *__restrict__ oldPos,
                typename vec4<T>::Type *vel,
                unsigned int deviceOffset, unsigned int deviceNumBodies,
                float deltaTime, float damping, int numTiles)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies)
    {
        return;
    }

    typename vec4<T>::Type position = oldPos[deviceOffset + index];

    typename vec3<T>::Type accel = computeBodyAccel<T>(position,
                                                       oldPos,
                                                       numTiles);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    typename vec4<T>::Type velocity = vel[deviceOffset + index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[deviceOffset + index] = position;
    vel[deviceOffset + index]    = velocity;
}

template <typename T>
void integrateNbodySystem(DeviceData<T> *deviceData,
                          cudaGraphicsResource **pgres,
                          unsigned int currentRead,
                          float deltaTime,
                          float damping,
                          unsigned int numBodies,
                          unsigned int numDevices,
                          int blockSize,
                          bool bUsePBO)
{
    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[currentRead], cudaGraphicsMapFlagsReadOnly));
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres[1-currentRead], cudaGraphicsMapFlagsWriteDiscard));
        checkCudaErrors(cudaGraphicsMapResources(2, pgres, 0));
        size_t bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[currentRead]), &bytes, pgres[currentRead]));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&(deviceData[0].dPos[1-currentRead]), &bytes, pgres[1-currentRead]));
    }

    for (unsigned int dev = 0; dev != numDevices; dev++)
    {
        if (numDevices > 1)
        {
            cudaSetDevice(dev);
        }

        int numBlocks = (deviceData[dev].numBodies + blockSize-1) / blockSize;
        int numTiles = (numBodies + blockSize - 1) / blockSize;
        int sharedMemSize = blockSize * 4 * sizeof(T); // 4 floats for pos

//////
//SSM design, do clean
	SSM_CLEAN<<<SSM_NUM_STREAM_MULTIPROCESSOR, SSM_CLEAN_BLOCK_SIZE>>>();

	//printf("hello, done clean \n");

        integrateBodies<T><<< numBlocks, blockSize, sharedMemSize >>>
        ((typename vec4<T>::Type *)deviceData[dev].dPos[1-currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dPos[currentRead],
         (typename vec4<T>::Type *)deviceData[dev].dVel,
         deviceData[dev].offset, deviceData[dev].numBodies,
         deltaTime, damping, numTiles);

	printf("hello, done calculation \n");

        if (numDevices > 1)
        {
            checkCudaErrors(cudaEventRecord(deviceData[dev].event));
            // MJH: Hack on older driver versions to force kernel launches to flush!
            cudaStreamQuery(0);
        }

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    if (numDevices > 1)
    {
        for (unsigned int dev = 0; dev < numDevices; dev++)
        {
            checkCudaErrors(cudaEventSynchronize(deviceData[dev].event));
        }
    }

    if (bUsePBO)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(2, pgres, 0));
    }
}


// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(DeviceData<float> *deviceData,
                                          cudaGraphicsResource **pgres,
                                          unsigned int currentRead,
                                          float deltaTime,
                                          float damping,
                                          unsigned int numBodies,
                                          unsigned int numDevices,
                                          int blockSize,
                                          bool bUsePBO);

template void integrateNbodySystem<double>(DeviceData<double> *deviceData,
                                           cudaGraphicsResource **pgres,
                                           unsigned int currentRead,
                                           float deltaTime,
                                           float damping,
                                           unsigned int numBodies,
                                           unsigned int numDevices,
                                           int blockSize,
                                           bool bUsePBO);
