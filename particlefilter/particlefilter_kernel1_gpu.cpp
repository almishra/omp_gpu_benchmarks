#include "particlefilter.h"

void particlefilter_kernel1_gpu(double *weights, double *arrayX, double *arrayY,
                                double xe, double ye, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*4;
  long mem_from = sizeof(double)*2;
  long mem_alloc = 0;
  long mem_del = 0;
  long start, end;

#pragma omp target data map(weights[0:N], arrayX[0:N], arrayY[0:N])
  {
  start = get_time();
#pragma omp target teams distribute parallel for map(to: xe, ye) map(num_teams, num_threads)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    weights[i] = 1.0 / N;
    arrayX[i] = xe;
    arrayY[i] = ye;
  }
  end = get_time();
  }
  fprintf(fp, "particlefilter_kernel1_gpu,%ld,1,1,%d,%d,%lu,%lu,%lu,%lu,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from, 
          mem_del, N);
}
