#include "mm.h"

void mm_kernel_cpu(double (*A)[N2],
                   double (*B)[N3],
                   double (*C)[N3],
                   FILE    *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp parallel for
  for (int i=0; i<N1; i++) {
    for (int j=0; j<N3; j++) {
      if(i == 0 && j == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  long start = get_time();
#pragma omp parallel for
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k=0; k<N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "mm_kernel_cpu,%ld,0,1,%d,%d,0,0,0,0,3,%d,%d,%d\n",
          (end - start), num_teams, num_threads, N1, N2, N3);
}
