#include "laplace.h"

double kernel1_cpu(double (*A)[N], double (*Anew)[N], double err, FILE *fp) {
  long start = get_time();
#pragma omp parallel for reduction(max: err)
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      Anew[i][j] = 0.25 * (A[i][j+1] + A[i][j-1] + A[i-1][j] + A[i+1][j]);

      double val;
      if(Anew[i][j] > A[i][j]) val = Anew[i][j] - A[i][j];
      else val = A[i][j] - Anew[i][j];

      if(err < val)
        err = val;
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel1_cpu,%ld,0,1,0,0,0,0,2,%d,%d\n", 
          (end - start), M, N);
  return err;
}
