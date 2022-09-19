#include "laplace.h"

void kernel2_gpu(double (*A)[N], double (*Anew)[N], FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for (int i = 1; i < M-1; i++) {
    for (int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  long start = get_time();
#pragma omp target teams distribute parallel for
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      A[i][j] = Anew[i][j];      
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel2_gpu,%ld,1,1,%d,%d,0,0,0,0,2,%d,%d\n",
          (end - start), num_teams, num_threads, M, N);
}
