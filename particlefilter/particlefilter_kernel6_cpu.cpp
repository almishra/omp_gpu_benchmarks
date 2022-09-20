#include "particlefilter.h"

void particlefilter_kernel6_cpu(double *CDF, double *u, double *arrayX,
                                double *arrayY, double *xj, double *yj,
                                FILE *fp)
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
    int x = -1;
    for(int j=0; j<N; j++) {
      if(CDF[j] >= u[i]) {
        x = j;
        break;
      }
    }
    if(x == -1) x = N-1;

    xj[i] = arrayX[x];
    yj[i] = arrayY[x];
  }
  long end = get_time();
  fprintf(fp, "particlefilter_kernel6_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
          (end - start), num_teams, num_threads, N);
}
