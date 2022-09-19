#include "mm.h"

void mm_kernel_gpu_mem(double (*A)[N2],
                       double (*B)[N3],
                       double (*C)[N3],
                       FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for (int i=0; i<N1; i++) {
    for (int j=0; j<N3; j++) {
      if(i == 0 && j == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  long mem_to = sizeof(double)*(N1*N2 + N2*N3);
  long mem_from = sizeof(double)*(N1*N3);
  long mem_alloc = sizeof(double)*(N1*N3);
  long mem_del = sizeof(double)*(N1*N2 + N2*N3 + N1*N3);

  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(alloc: C[0:N1][0:N3]) \
                   map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
                   map(from: C[0:N1][0:N3])
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }

  long end = get_time();
  fprintf(fp, "mm_kernel_gpu_mem,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,3,%d,%d,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, N1, N2, N3);
}
