#include "laplace.h"

void kernel2_cpu(double (*A)[N], double (*Anew)[N], FILE *fp) {
  long start = get_time();
#pragma omp parallel for
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      A[i][j] = Anew[i][j];      
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel2_cpu,%ld,0,1,0,0,0,0,2,%d,%d\n", 
          (end - start), M, N);
}
