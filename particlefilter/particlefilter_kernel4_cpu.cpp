#include "particlefilter.h"

void particlefilter_kernel4_cpu(double *arrayX, double *arrayY, double *weights,
                                double &xe, double &ye, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;

  long start = get_time();
#pragma omp parallel for reduction(+:xe, ye)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    xe += arrayX[i] * weights[i];
    ye += arrayY[i] * weights[i];
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel4_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
