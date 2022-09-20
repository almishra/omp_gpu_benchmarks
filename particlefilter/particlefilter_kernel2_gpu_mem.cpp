#include "particlefilter.h"

double particlefilter_kernel2_gpu_mem(double *weights, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*(N+3);
  long mem_from = sizeof(double)*3;
  long mem_alloc = 0;
  long mem_del = sizeof(double)*N;
  double sumWeights = 0;

  long start = get_time();
#pragma omp target teams distribute parallel for reduction(+:sumWeights) \
                                    map(sumWeights) map(to:weights[0:N]) \
                                    map(num_teams, num_threads)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    sumWeights += weights[i];
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel2_gpu_mem,%ld,1,1,%d,%d,%lu,%lu,%lu,%lu,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from, 
          mem_del, N);

  return sumWeights;
}
