#include "laplace.h"

double kernel1_cpu_collapse(double (*A)[N], double (*Anew)[N], double err, FILE *fp) {
  long start = get_time();
#pragma omp parallel for collapse(2) reduction(max: err)
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

  fprintf(fp, "laplace_kernel1_cpu_collapse,0,0,0,0,%d,%d,%ld\n", M, N, (end - start));

  return err;
}
