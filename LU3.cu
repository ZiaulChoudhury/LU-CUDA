#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<sys/time.h>
#include <cooperative_groups.h>
#define N 8
#define T 512
float c[N][N];

namespace cg = cooperative_groups;
float l[T][T];
float u[T][T];

void mylu(float *a) {
    int n = T;
    int i,j,k,p;
    float sum;
    for(k=0;k<n;k++)
    {
        u[k][k]=1;
        for(i=k;i<n;i++)
        {
            sum=0;
            for(p=0;p<k;p++)
                sum+=l[i][p]*u[p][k];
            l[i][k]=a[i*T + k]-sum;
        }

        for(j=k+1;j<n;j++)
        {
            sum=0;
            for(p=0;p<k;p++)
                sum+=l[k][p]*u[p][j];
            u[k][j]=(a[k*T + j]-sum)/l[k][k];
        }
    }

}

void multiply(float l[N][N], float u[N][N])
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += l[i][k] * u[k][j];
            }
        }
    }
}

__global__ void decompose(float *A, float *pivots, int iteration)
{
	int blockID = blockIdx.x;
	int threadId = threadIdx.x;
	float p = 0;
	if(blockID >= iteration){
		p = A[blockIdx.x * N + iteration - 1]/A[(iteration - 1)*N + iteration - 1];
		A[blockID*N + threadId] -= p * A[(iteration-1)*N + threadId];
		A[blockID*N + iteration-1] = p;
	}
}
__global__ void TSolve(float *L, float *A, float *U01)
{
	int tid = threadIdx.x;
	float sum = 0;
	int i,j;
	for(i=0;i<N;i++){
		for(j=0;j<i;j++)
			sum += U01[j*N + tid]*L[j*tid+ j];
	U01[j*N + tid] = (A[i*N + tid] - sum)/L[j*N + tid]; 		
	}	
}
__global__ void TSolve2(float *U, float *A, float *L10)
{
	int tid = threadIdx.x;
	float sum = 0;
        int i,j;
       	for(i=N-1;i>=0;i--){
                for(j=i;j<N;j++)
                        sum += L10[j*N + tid]*U[j*tid+ j];
        L10[j*N + tid] = (A[i*N + tid] - sum)/U[j*N + tid];
        }
}

void printLU(){
 	printf("\n ---------------- L VALUES ------------------- \n");
        for(int i=0;i<T;i++){
        for(int j=0;j<T;j++)
                printf(" %6.2f ", l[i][j]);
        printf("\n");
        }
        printf("\n-----------------------------------------------\n");

        printf("\n ---------------- U VALUES ------------------- \n");
        for(int i=0;i<T;i++){
        for(int j=0;j<T;j++)
                printf(" %6.2f ", u[i][j]);
        printf("\n");
        }
        printf("\n-----------------------------------------------\n");
}
int main(int argc, char *argv[]){ 
        float *A;
	float *pivots;
        float *dev_a, *dev_pivots;
        float *dev_u00;
        float *dev_l00;
        float *dev_l10;
        float *dev_u01;
	int *devItr;
        A=(float *)malloc(sizeof(float)*T*T);
	cudaEvent_t start, stop;
        float time;
        float totalTime=0;
	
	cudaMalloc ( (void**)&dev_a, T*T* sizeof (float) );
	cudaMalloc ( (void**)&dev_u00, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_l00, N*N* sizeof (float) );
	cudaMalloc ( (void**)&dev_l10, (T-N)*N*sizeof (float) );
	cudaMalloc ( (void**)&dev_u01, N*(T-N)*sizeof (float) );
        cudaMalloc ( (void**)&dev_pivots, N*sizeof (float) );
        cudaMalloc ( (void**)&devItr, sizeof (int) );

        pivots=(float *)malloc(sizeof(float)*N);
	for(int i=0;i<T*T;i++)
		A[i] = (float)(rand()%100);;

	cudaMemcpy(dev_a, A, T*T*sizeof(float), cudaMemcpyHostToDevice);
	
	for(int i=0;i<T;i++){
		for(int j=0;j<T;j++)
		printf(" %6.2f ", A[i*T + j]);
	printf("\n");
	}

	mylu(A);
	
	printf("\n\n");

	for(int i=1;i<N;i++)
		pivots[i] = A[(i)*N]/A[0];

	cudaMemcpy(dev_pivots, pivots, N*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
        cudaEventCreate(&stop);	
	cudaEventRecord(start, 0);	

	for(int m = 0; m<T/N; m++) {	
	for(int i=1;i<N;i++) {
		decompose<<<N,N>>>(dev_a,dev_pivots,i);
		cudaThreadSynchronize();	
		cudaMemcpy(A, dev_a, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	}	
	cudaEventRecord(stop, 0);
        cudaEventElapsedTime(&time, start, stop);
        totalTime += time;

	
	float L[N][N] ={0};
	float U[N][N] = {0};
	for(int i=1;i<N;i++)
		for(int j=0;j<i;j++)
			L[i][j] = A[i*N + j];

	for(int i=0;i<N;i++)
		L[i][i] = 1;

	for(int i=0;i<N;i++)
                for(int j=i;j<N;j++)
                        U[i][j] = A[i*N + j];
	
	cudaMemcpy(dev_u00, U, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_l00, L, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	TSolve<<<1,(T-N)>>>(dev_l00,  dev_a, dev_u01);
	TSolve2<<<1,(T-N)>>>(dev_u00, dev_a, dev_l10);
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();	
	cudaEventElapsedTime(&time, start, stop);
	totalTime += time;
	}
		
	printf("\n \n GPU kernel execution time = %f ms\n",totalTime);
	
}
