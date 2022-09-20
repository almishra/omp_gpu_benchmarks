#include "particlefilter.h"

double particlefilter_kernel2_cpu(double *weights, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  double sumWeights = 0;

  long start = get_time();
#pragma omp parallel for reduction(+:sumWeights)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    sumWeights += weights[i];
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel2_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);

  return sumWeights;
}
