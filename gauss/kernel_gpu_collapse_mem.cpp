#include "gauss.h"

int kernel_gpu_collapse_mem(double (*mat)[N], FILE *fp) {
  int diff = 0;
  long mem_to = sizeof(double)*N*N;
  long mem_from = sizeof(double)*N*N;
  long mem_alloc = 0;
  long mem_delete = 0;
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

  fprintf(fp, "kernel_gpu_collapse_mem,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_delete, N, (end - start));

  return diff;
}

