#include <stdlib.h> 
#include <string.h> 
#include <stdio.h> 
#include <mpi.h>
#include <math.h>

void MergeSort (int n, int *a, int *b);
double log2(double n);
double pow(double m, double n);

int HQuicksort(int *Abuf, int *lenA,  int *list, int N, MPI_Comm comm) {
/*--------------------    
 Most of  the lab2  assignment will consist  of writing  this function.
 Right now,  it is set  to just print ``nothing  done" and exit.  It is
 provided so you can compile and run the pgm - but the numbers will not
 be sorted.
-------------------- */

//*Abuf=Loc_dat
//*lenA means the size of Abuf
//list includes all pivot
//N means total number
 /* 1. each node use list[i], split into two pieces
  2. processes send Ai-high and receive Ai-low
  3. sort again(merge)
  4. repeat*/
  // myid and number of processes
  int myid,nprocs;
  MPI_Comm_rank(comm,&myid);
  MPI_Comm_size(comm,&nprocs);
  int ngroups;
  //temp store temp of Abuf
  int *temp;
  temp = (int *) malloc(N*sizeof(int));
  memcpy(temp, Abuf, N*sizeof(int));
  int m = 0, n, index, data, bit_change, bith_change;
  int target, count_number,index2;
  MPI_Status status[2];
  MPI_Request req[2];

  //size means the maximum steps need to be done
  int size = (int) log2((double) nprocs);
  //i means the number of steps
  for(int i = 1; i <= size; i++)
  {
    //j means maybe maximum existing j groups in each step
    for(int j = 0; j < (int) pow(2.0,(double)i - 1); j++)
    {
      //ngroups means the number of groups in each step
      //find the specific number of list and then sort, split them
      ngroups = myid >> (size+1-i);//for example, when size=3 and i=1, should be 1 group(ngroup+1)
      //in each group
      if(ngroups == j)
      {
        //sort  
        //index means the number smaller than target
        index = 0;
        for(int k = 0; k < *lenA; k++)
        {
          if(Abuf[k] <= list[m + j])
          {
            data = Abuf[k];
            Abuf[k] = Abuf[index];
            Abuf[index] = data;
            index++;
          }
        }
        //split through using index
        bit_change = size - i;
        //send and receive small number when == 0
        target = (int) pow(2.0,(double) bit_change);
        bith_change = (myid & target) >> bit_change;
        if(bith_change == 0)
        {
          //send large number
          MPI_Isend(&Abuf[index], *lenA-index, MPI_INT, myid + target, 0, comm, &req[0]);
          //receive small number
          MPI_Irecv(temp, N, MPI_INT, myid + target, 0, comm, &req[1]);
          MPI_Waitall(2, req, status);
          //count_number save the number that receive
          MPI_Get_count(&status[1], MPI_INT, &count_number);
          MPI_Barrier(comm);
          //update *lenA
          *lenA = index + count_number;
          //update Abuf
          for(n = 0; n < count_number; n++)
            Abuf[index + n] = temp[n];
        }
        else if(bith_change == 1)
        {
          //send small number
          MPI_Isend(&Abuf[0], index, MPI_INT, myid - target, 0, comm, &req[0]);
          //receive large number
          MPI_Irecv(temp, N, MPI_INT, myid - target, 0, comm, &req[1]);
          MPI_Waitall(2, req, status);
          //count_number save the number that receive
          MPI_Get_count(&status[1], MPI_INT, &count_number);
          MPI_Barrier(comm);
          //because *lenA may change
          index2 = *lenA - index;
          //update *lenA
          *lenA = *lenA - index + count_number;
          //update Abuf
          for(n = index; n < index + index2; n++)
            Abuf[n-index] = Abuf[n];
          for(n = 0; n < count_number; n++)
            Abuf[index2 + n] = temp[n];
        }
      }
    }
    MergeSort(*lenA, Abuf, temp);
    m += (int) pow(2.0,(double) (i - 1));
  }
  //free(temp);
  return(0);
}
