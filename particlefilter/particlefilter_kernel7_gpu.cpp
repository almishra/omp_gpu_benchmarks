#include "particlefilter.h"

void particlefilter_kernel7_gpu(double *weights, double *arrayX, double *arrayY,
                                double *xj, double *yj, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*2;
  long mem_from = sizeof(double)*2;
  long mem_alloc = 0;
  long mem_del = 0;
  long start, end;

#pragma omp target data map(to: xj[0:N], yj[0:N]) \
                        map(from: arrayX[0:N], arrayY[0:N], weights[0:N])
  {
  start = get_time();
  // reassign arrayX and arrayY and weights
#pragma omp target teams distribute parallel for map(num_teams, num_threads) 
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];
    weights[i] = 1.0 / N;
  }
  end = get_time();
  }
  fprintf(fp, "particlefilter_kernel7_gpu,%ld,1,1,%d,%d,%lu,%lu,%lu,%lu,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from, 
          mem_del, N);
}
