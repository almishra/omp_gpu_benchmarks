#include "mm.h"

void multiply_gpu_collapse2(double (*A)[N2],
                            double (*B)[N3],
                            double (*C)[N3],
                            FILE *fp)
{
  long mem_to = 0;
  long mem_from = 0;
  long mem_alloc = 0;
  long mem_del = 0;

#ifdef MEMCPY
  mem_to = sizeof(double)*(N1*N2 + N2*N3);
  mem_alloc = sizeof(double)*(N1*N3);
  mem_from = sizeof(double)*(N1*N3);
  mem_del = sizeof(double)*(N1*N2 + N2*N3);
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
                              map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
                             map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_gpu_col2,2,%ld,%ld,%ld,%ld,%d,%d,%d,%ld\n",
          mem_to, mem_alloc, mem_from, mem_del, N1, N2, N3, end - start);
}
