#include "covariance.h"

double covariance_kernel1_cpu(double *X, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  double sum = 0;

  long start = get_time();
#pragma omp parallel for reduction(+: sum)
  for (int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    sum += X[i];
  }
  long end = get_time();

  double mean = sum / (double)N;

  fprintf(fp, "covariance_kernel1_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);

  return mean;
}
