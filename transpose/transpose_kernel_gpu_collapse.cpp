#include "transpose.h"

void transpose_kernel_gpu_collapse(double (*A)[N2], double (*B)[N1], FILE *fp)
{
  int num_threads = 1;
  int num_teams = 1;
  long mem_to = sizeof(int)*2;
  long mem_from = sizeof(int)*2;
  long mem_alloc = 0;
  long mem_del = 0;
  long start, end;

#pragma omp target data map(to: A[0:N1][0:N2]) map(from: B[0:N2][0:N1])
  {
  start = get_time();
#pragma omp target teams distribute parallel for collapse(2) map(num_teams, num_threads)
  for(int i=0; i<N2; i++) {
    for(int j=0; j<N1; j++) {
      if(i == 0 && j == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      B[i][j] = A[j][i];
    }
  }
  end = get_time();
  }

  fprintf(fp, "transpose_kernel_gpu_collapse,%ld,1,2,%d,%d,%ld,%ld,%ld,%ld,2,%d,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, N1, N2);
}
