#include "transpose.h"

void transpose_kernel_cpu_collapse(double (*A)[N2], double (*B)[N1], FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;

  long start = get_time();
#pragma omp parallel for collapse(2)
  for(int i=0; i<N2; i++) {
    for(int j=0; j<N1; j++) {
      if(i == 0 && j == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      B[i][j] = A[j][i];
    }
  }
  long end = get_time();

  fprintf(fp, "transpose_kernel_cpu_collapse,%ld,0,2,%d,%d,0,0,0,0,2,%d,%d\n",
          (end - start), num_teams, num_threads, N1, N2);
}
