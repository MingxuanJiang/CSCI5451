#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>


/*-------------------- POSIX-compliant timer in seconds */
double wctime() 
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}


