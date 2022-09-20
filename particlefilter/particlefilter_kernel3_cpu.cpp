#include "particlefilter.h"

void particlefilter_kernel3_cpu(double *weights, double sumWeights, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;

  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    weights[i] = weights[i] / sumWeights;
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel3_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
