#include "gauss.h"

int kernel_gpu_collapse(double (*mat)[N], FILE *fp) {
  int diff = 0;
  long start = get_time();
#pragma omp target teams distribute parallel for collapse(2) reduction(+:diff) \
                   map(diff)
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

  fprintf(fp, "gauss_kernel_gpu_collapse,%ld,1,2,%lu,0,%lu,0,1,%d\n",
          (end - start), sizeof(int), sizeof(int), N);

  return diff;
}

