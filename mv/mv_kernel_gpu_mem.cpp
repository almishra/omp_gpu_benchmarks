#include "mv.h"

void mv_kernel_gpu_mem(double (*A)[N2], double *B, double *C, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
  long mem_to = sizeof(double)*(N1*N2 + N2) + sizeof(int)*2;
  long mem_from = sizeof(double)*N1 + sizeof(int)*2;
  long mem_alloc = sizeof(double)*N1;
  long mem_del = sizeof(double)*(N1*N2 + N2);

  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(to: A[0:N1][0:N2], B[0:N2]) map(from: C[0:N1]) \
                   map(num_teams, num_threads)
  for(int i=0; i<N1; i++) {
      if(i == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      double sum = 0.0;
    for(int j=0; j<N2; j++) {
        sum = sum + A[i][j] * B[j];
    }
      C[i] = sum;
  }

  long end = get_time();
  fprintf(fp, "mv_kernel_gpu_mem,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,2,%d,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, N1, N2);
}
