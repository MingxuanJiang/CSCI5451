#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>

/*-------------------- POSIX-compliant timer in seconds */
// calculate the time with function wctime()
double wctime() 
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

// initialize the length of vector
#define NN 524288

// calculate the form
__global__ void saxpy_par(int N, float a, float *A, float *B){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
       B[i] = a * A[i] + B[i]; 
    }
}

float saxpy_check(int n, float a, float *x, float *y, float *z) {
    // a, x, y == original data for saxpy
    // z = result found -- with which to compare.
    float s=0.0, t = 0.0;
    for (int i=0; i<n; i++) {
        y[i] += a * x[i] ;
        s += (y[i] - z[i])*(y[i] - z[i]);
        t += z[i]*z[i];
    }
    if (t == 0.0) return(-1);
    else
    return(sqrt(s/t));
}

int main(){
    float *x_d, *y_d, *z_d;    // vector x, y, z in GPUs
    float *x, *y, *z;          // vector x, y, z in CPUs
    float a = 1, Mflops, err;
    double t1, t2;             // time
    int N, i, vecLen, iter;
    int MatSize;               // the size of vector
    //-------------------- set dimension N
    N = NN;
    MatSize = N*sizeof(float);
    //-------------------- allocate on cpu
    x = (float *)malloc(MatSize);        
    y = (float *)malloc(MatSize);        
    z = (float *)malloc(MatSize);    
    if ((x==NULL) | (y==NULL) | (z==NULL) ) 
        exit(1); 
    //-------------------- allocate on GPU
    if (cudaMalloc((void **) &x_d, MatSize) != cudaSuccess) 
        exit(2);	      
    if (cudaMalloc((void **) &y_d, MatSize) != cudaSuccess) 
        exit(3);	      
    if (cudaMalloc((void **) &z_d, MatSize) != cudaSuccess) 
        exit(4);
    
    //set vector x,y with random numbers
    for(i = 0 ; i < N ; i ++)	{
        x[i]  = (float) rand() / (float) rand();
        y[i]  = (float) rand() / (float) rand();
    }
    int NITER = 100;
    a = a/(float) NITER;
    for(vecLen = 1024; vecLen <= N; vecLen *= 2){
        //-------------------- copy matrices x,y to GPU memory
        cudaMemcpy(x_d, x, sizeof(float)*vecLen, cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y, sizeof(float)*vecLen, cudaMemcpyHostToDevice);
        // set dimension of block and grid
        dim3 dimBlock = dim3(1024);
        dim3 dimGrid = dim3(vecLen/1024);
        t1 = wctime(); // record time here
        for(iter = 0;iter<NITER;iter++){
            saxpy_par<<<dimGrid,dimBlock>>>(vecLen,a,x_d,y_d);            
        }
        t2 = wctime(); // record exit time here
        cudaMemcpy(z, y_d, sizeof(float)*vecLen, cudaMemcpyDeviceToHost);
        Mflops = 2*vecLen*NITER*1E-6/(t2-t1);
        //check error
        err = 0.0;
        err = saxpy_check(vecLen,1,x,y,z);
        // print results for this vecLen...
        printf("** vecLen = %d, Mflops = %.2f, err = %.2e\n",vecLen,Mflops,err);
    }
    //-------------------- Free Host arrays
    free(x); 
    free(y);
    free(z);
    //-------------------- Free GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);	
    return 0;
}




