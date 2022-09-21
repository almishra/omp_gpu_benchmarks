#include "laplace.h"

double kernel1_gpu(double (*A)[N], double (*Anew)[N], double err, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp target teams distribute parallel for reduction(max: err) \
                                    map(err, num_teams, num_threads)
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      Anew[i][j] = 0.25 * (A[i][j+1] + A[i][j-1] + A[i-1][j] + A[i+1][j]);

      double val;
      if(Anew[i][j] > A[i][j]) val = Anew[i][j] - A[i][j];
      else val = A[i][j] - Anew[i][j];

      if(err < val)
        err = val;
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel1_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,2,%d,%d\n",
          (end - start), num_teams, num_threads, sizeof(double)+2*sizeof(int),
          sizeof(double)+2*sizeof(int), M, N);
  return err;
}
