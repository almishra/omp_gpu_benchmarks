#include "particlefilter.h"

void particlefilter_kernel5_cpu(double *u, double u1, FILE *fp)
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
    u[i] = u1 + i/((double)(N));
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel5_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
