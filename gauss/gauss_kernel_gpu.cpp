#include "gauss.h"

float kernel_gpu(double (*mat)[N], FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for (int i = 1; i < N-1; i++) {
    for (int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  float diff = 0;
  long start = get_time();
#pragma omp target teams distribute parallel for reduction(+:diff) map(diff)
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

  fprintf(fp, "gauss_kernel_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,1,%d\n", 
          (end - start), num_teams, num_threads, sizeof(int), sizeof(int), N);

  return diff;
}

