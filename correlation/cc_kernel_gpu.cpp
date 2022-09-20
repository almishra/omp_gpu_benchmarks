#include "cc.h"

double cc_kernel_gpu(double *X, double *Y, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(double)*7;
  long mem_from = sizeof(double)*7;
  long mem_alloc = 0;
  long mem_del = 0;
  double sum_X = 0, sum_Y = 0, sum_XY = 0;
  double squareSum_X = 0, squareSum_Y = 0;
  long start, end;

#pragma omp target data map(to: X[0:N], Y[0:N])
  {
  start = get_time();
#pragma omp target teams distribute parallel for reduction(+: sum_X, sum_Y, sum_XY, squareSum_X, squareSum_Y) \
                                    map(sum_X, sum_Y, sum_XY, squareSum_X, squareSum_Y) \
                                    map(num_teams, num_threads)
  for (int i=0; i<N; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    sum_X += X[i];
    sum_Y += Y[i];
    sum_XY += X[i] * Y[i];
    squareSum_X += X[i] * X[i];
    squareSum_Y += Y[i] * Y[i];
  }
  end = get_time();
  }
  double corr = (double)(N * sum_XY - sum_X * sum_Y) / 
                sqrt((N * squareSum_X - sum_X * sum_X) * 
                     (N * squareSum_Y - sum_Y * sum_Y));
  fprintf(fp, "cc_kernel_gpu,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, N);

  return corr;
}
