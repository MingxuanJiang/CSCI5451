#include <stdlib.h>
#include <string.h> 
#include <stdio.h>
#include <mpi.h>

/*--------------- Gaussian elimination -- Pipelined version */
int pipe_ge(double *AA, int n,  MPI_Comm comm){
  /* Pipelined Gaussian Elimination  
     AA = pointer to matrix and right-hand side. AA is a one-dimensional 
	  array holding a matrix of size n*(n+1)-- row-wise. The
          right-hand side occupies the last column. 
     n  = dimension of problem
     On return Gaussian elimination has been applied and only upper triangular
     part is relevant -- back-solve should be called to get solution x.  
     *----------------------------------------------------------------------*/
  int nprocs, myid;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  //MPI_Request req[2];
  MPI_Status status;
  double piv;
  int kloc, iloc;
  //initialize temp
  double *temp;
  temp = (double*)malloc(sizeof(double)*n);
  
  for (int k = 0; k < n - 1; k++)
  {
    kloc = k / nprocs;
    //if row k is in proc myid
    if((k % nprocs) == myid)
    {
      //send row k to proc. South
      MPI_Send(&AA[kloc*(n+1)], n + 1, MPI_DOUBLE, (myid+1)%nprocs, 0, comm);
      MPI_Recv(&temp[0], n+1, MPI_DOUBLE, (myid-1)%nprocs, 0, comm, &status);
    }      
    else
    {
      //receieve row k from proc. North
      MPI_Recv(&temp[0], n + 1, MPI_DOUBLE, (myid-1)%nprocs, 0, comm, &status);
      //send row k to south
      if(k < n-1)
        MPI_Send(&temp[0], n + 1, MPI_DOUBLE, (myid+1)%nprocs, 0, comm);
    }         
    MPI_Barrier(comm);
    //compute piv and elimination
    for(int i = k + 1; i < n; i++)
    {
      iloc = i / nprocs;
      if((i % nprocs) == myid)
      {
        piv = AA[iloc*(n+1)+k] / temp[k];
      for(int j = k + 1; j < n+1; j++)
        AA[iloc*(n+1)+j] = AA[iloc*(n+1)+j]-piv*temp[j];
      }      
    }
  }
  return(0);
}
