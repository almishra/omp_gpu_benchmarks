#include "mm.h"

void mm_kernel_gpu_collapse(double (*A)[N2],
                            double (*B)[N3],
                            double (*C)[N3],
                            FILE *fp)
{
  long start = get_time();
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "mm_kernel_gpu_collapse,%ld,1,2,0,0,0,0,3,%d,%d,%d\n",
          (end - start), N1, N2, N3);
}