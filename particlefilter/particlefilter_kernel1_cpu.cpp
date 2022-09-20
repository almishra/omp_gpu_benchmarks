#include "particlefilter.h"

void particlefilter_kernel1_cpu(double *weights, double *arrayX, double *arrayY,
                                double xe, double ye, FILE *fp)
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
    weights[i] = 1.0 / N;
    arrayX[i] = xe;
    arrayY[i] = ye;
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel1_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
