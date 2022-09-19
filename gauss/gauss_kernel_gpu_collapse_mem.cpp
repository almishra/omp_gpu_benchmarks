#include "gauss.h"

float kernel_gpu_collapse_mem(double (*mat)[N], FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp target teams distribute parallel for collapse(2) map(num_teams, num_threads)
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  float diff = 0;
  long mem_to = sizeof(double)*N*N + sizeof(int);
  long mem_from = sizeof(double)*N*N + sizeof(int);
  long mem_alloc = 0;
  long mem_delete = sizeof(double)*N*N;
  long start = get_time();
#pragma omp target teams distribute parallel for collapse(2) reduction(+:diff) \
                   map(mat[0:N][0:N])
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      const float temp = mat[i][j];
      mat[i][j] = 0.2f * (
          mat[i][j]
          + mat[i][j-1]
          + mat[i-1][j]
          + mat[i][j+1]
          + mat[i+1][j]
          );

      float x = mat[i][j] - temp;
      if(x < 0) x *= -1;
      diff += x;
    }
  }
  long end = get_time();
#pragma omp target exit data map(delete: mat[0:N][0:N])

  fprintf(fp, "gauss_kernel_gpu_collapse_mem,%ld,1,2,%d,%d,%ld,%ld,%ld,%ld,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_delete, N);

  return diff;
}

