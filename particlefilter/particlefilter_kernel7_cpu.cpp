#include "particlefilter.h"

void particlefilter_kernel7_cpu(double *weights, double *arrayX, double *arrayY,
                                double *xj, double *yj, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;

  long start = get_time();
  // reassign arrayX and arrayY and weights
#pragma omp parallel for
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];
    weights[i] = 1.0 / N;
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel7_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
