#include <stdlib.h> 
#include <stdio.h>
#include <mpi.h>
int back_solve(double *A, int n, int nloc, double *x, MPI_Comm comm){
  /*--------------------
  / upper triangular solution 
  / A = pointer to an n * (n+1) matrix. Only the upper triangular part of
        A is used. The right-hand side occupies the last column of A.
    The column version of the algorithm is implemented. 
  */
  /*-------------------- NOTE: THIS JUST SETS X = RHS AND RETURNS */
  /* it is provided so that you can see how to access cerain elements of
     the matrix stored as a one-dimensional array                 */
  int k, id,  nprocs, np=n+1, myid, kloc, iloc;
  double t;
  t = 0;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &myid); 
  /*--------------- back-solve loop          */
  for (k=n-1; k>=0; k--) {
    id = (k % nprocs);
    if (id == myid) {
      kloc = (k-id) / nprocs;
      t = A[kloc*np+n] / A[kloc*np+k];
/*-------------------- set x=rhs on return */ 
      x[kloc] = t;
    }
    //broadcast to all processors
    MPI_Bcast(&t, 1, MPI_DOUBLE, id, comm);
    MPI_Barrier(comm);
    //substract multiple of (relevant) part of columm k from rhs
    for(int i = 0; i < k; i++)
    {
      id = (i % nprocs);
      if(id == myid)
      {
        iloc = (i - id) / nprocs;
        A[iloc*(n+1)+n] = A[iloc*(n+1)+n] - t*A[iloc*(n+1)+k];
      }      
    }
        
  }
  printf (" ------  end back-solve  in PE %d   ---------\n \n",myid);
  return(0);
}
