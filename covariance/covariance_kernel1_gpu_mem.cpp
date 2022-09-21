#include "covariance.h"

double covariance_kernel1_gpu_mem(double *X, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*(3+N);
  long mem_from = sizeof(double)*3;
  long mem_alloc = 0;
  long mem_del = sizeof(double)*N;
  double sum = 0;
  long start, end;

  start = get_time();
#pragma omp target teams distribute parallel for reduction(+: sum) \
                                    map(to: X[0:N]) \
                                    map(sum, num_teams, num_threads)
  for (int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    sum += X[i];
  }
  end = get_time();

  double mean = sum / (double)N;

  fprintf(fp, "covariance_kernel1_gpu_mem,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, N);

  return mean;
}
