#include "particlefilter.h"

void particlefilter_kernel4_gpu(double *arrayX, double *arrayY, double *weights,
                                double &xe, double &ye, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*4;
  long mem_from = sizeof(double)*4;
  long mem_alloc = 0;
  long mem_del = 0;
  long start, end;

#pragma omp target data map(to: arrayX[0:N], arrayY[0:N], weights[0:N])
  {
  start = get_time();
#pragma omp target teams distribute parallel for reduction(+:xe, ye) \
                                    map(xe, ye, num_teams, num_threads)
  for(int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    xe += arrayX[i] * weights[i];
    ye += arrayY[i] * weights[i];
  }
  end = get_time();
  }
  fprintf(fp, "particlefilter_kernel4_gpu,%ld,0,1,%d,%d,%lu,%lu,%lu,%lu,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from, 
          mem_del, N);
}
