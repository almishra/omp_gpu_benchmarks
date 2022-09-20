#include "particlefilter.h"

void particlefilter_kernel3_gpu(double *weights, double sumWeights, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*3;
  long mem_from = sizeof(double)*2;
  long mem_alloc = 0;
  long mem_del = 0;
  long start, end;

#pragma omp target data map(weights[0:N])
  {
  start = get_time();
#pragma omp target teams distribute parallel for map(to: sumWeights) \
                                    map(num_teams, num_threads)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    weights[i] = weights[i]/sumWeights;
  }
  end = get_time();
  }
  fprintf(fp, "particlefilter_kernel3_gpu,%ld,1,1,%d,%d,%lu,%lu,%lu,%lu,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from, 
          mem_del, N);
}
