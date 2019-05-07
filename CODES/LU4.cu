#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<sys/time.h>
#include <cooperative_groups.h>
#define N 128
#define ITER 2

__global__ void decompose(float *A, float *pivots, int iteration)
{
	int blockID = blockIdx.x;
	int threadId = threadIdx.x;
	int bid;
	float p = 0;
	int i;
	for(i=0;i<ITER;i++){
		bid = i*(N/ITER) + blockID;
		if(bid >= iteration){
			p = A[bid * N + iteration - 1]/A[(iteration - 1)*N + iteration - 1];
			A[bid*N + threadId] -= p * A[(iteration-1)*N + threadId];
			A[bid*N + iteration-1] = p;
		}
	}
}

void printA(float *A){
	for(int i=0;i<N;i++){
                        for(int j=0;j<N;j++)
                                printf(" %8.2f ", A[i*N + j]);
                printf("\n");
                }

}
int main(int argc, char *argv[]){ 
        float *A;
	float *pivots;
        float *dev_a, *dev_pivots;
	int *devItr;
        A=(float *)malloc(sizeof(float)*N*N);
	cudaEvent_t start, stop;
        float time;
        float totalTime=0;
	
	cudaMalloc ( (void**)&dev_a, N*N* sizeof (float) );
        cudaMalloc ( (void**)&dev_pivots, N*sizeof (float) );
        cudaMalloc ( (void**)&devItr, sizeof (int) );

        pivots=(float *)malloc(sizeof(float)*N);
	for(int i=0;i<N*N;i++)
		A[i] = (float)(rand()%100);;

	cudaMemcpy(dev_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	/*for(int i=0;i<N;i++){
		for(int j=0;j<N;j++)
		printf(" %6.2f ", A[i*N + j]);
	printf("\n");
	}*/

	//printf("\n\n");

	for(int i=1;i<N;i++)
		pivots[i] = A[(i)*N]/A[0];

	cudaMemcpy(dev_pivots, pivots, N*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	for(int i=1;i<N;i++) {
	cudaEventRecord(start, 0);
	decompose<<<N/ITER,N>>>(dev_a,dev_pivots,i);	
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();	
	//printf("\n");
	cudaMemcpy(A, dev_a, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;
	}
	//printA(A);	
        printf("\n \n GPU kernel execution time = %f ms\n",totalTime);
	
}
