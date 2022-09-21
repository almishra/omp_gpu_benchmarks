#include "laplace.h"

void kernel2_cpu(double (*A)[N], double (*Anew)[N], FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp parallel for
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      A[i][j] = Anew[i][j];      
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel2_cpu,%ld,0,1,%d,%d,0,0,0,0,2,%d,%d\n",
          (end - start), num_teams, num_threads, M, N);
}
