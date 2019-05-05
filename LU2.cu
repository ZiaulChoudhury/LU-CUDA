#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<sys/time.h>
#include <cooperative_groups.h>
#define N 16

namespace cg = cooperative_groups;
extern "C" __global__ void singleKerneldecompose(float *A, float *pivots)
{
        int blockID = blockIdx.x;
        int threadId = threadIdx.x;
        float p = 0;
	cg::grid_group grid = cg::this_grid();
       

	for(int i=1;i<N;i++){
	if(blockID >= i){
               	p = A[blockIdx.x * N + i - 1]/A[(i - 1)*N + i - 1];
                if(threadIdx.x == 0)
                        printf(" L[%d] =  %6.2f ", blockIdx.x, p);
                A[blockID*N + threadId] -= p * A[(i-1)*N + threadId];
        	}
	 cg::sync(grid);
	 if(blockID ==0 && threadId == 0)
		printf("\n");
	 }
}


int main(int argc, char *argv[]){ 
        float *A;
	float *pivots;
	cudaMallocManaged(reinterpret_cast<void **>(&A), sizeof(float) * (N * N));
	cudaMallocManaged(reinterpret_cast<void **>(&pivots), sizeof(float) * (N));
	cudaEvent_t start, stop;
        float time;

	for(int i=0;i<N*N;i++)
		A[i] = (float)(rand()%100);;
	
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++)
		printf(" %6.2f ", A[i*N + j]);
	printf("\n");
	}

	cudaEventCreate(&start);
        cudaEventCreate(&stop);

	printf("\n\n");
        int sMemSize = sizeof(double) * N;	
	void *kernelArgs[]={(void*)&A, (void*)&pivots};
	cudaEventRecord(start, 0);
	cudaLaunchCooperativeKernel((void *)singleKerneldecompose,
                                              N, N, kernelArgs,
                                              sMemSize, NULL);
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("\n\n");
	for(int i=0;i<N;i++){
                for(int j=0;j<N;j++)
                printf(" %6.2f ", A[i*N + j]);
        printf("\n");
        }
	
	printf("\n \n GPU kernel execution time = %f ms\n",time);
	
	
}
