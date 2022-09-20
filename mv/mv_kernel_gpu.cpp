#include "mv.h"

void mv_kernel_gpu(double (*A)[N2], double *B, double *C, FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;

  long start = get_time();
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for(int i=0; i<N1; i++) {
    if(i == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    double sum = 0.0;
    for(int j=0; j<N2; j++) {
      sum += A[i][j] * B[j];
    }
    C[i] = sum;
  }
  long end = get_time();
  fprintf(fp, "mv_kernel_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,2,%d,%d\n", (end - start),
          num_teams, num_threads, 2*sizeof(int), 2*sizeof(int), N1, N2);
}
