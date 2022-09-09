#include "laplace.h"

void kernel2_gpu_mem(double (*A)[N], double (*Anew)[N], FILE *fp) {
  long mem_to = sizeof(double)*M*N;
  long mem_from = sizeof(double)*M*N;
  long mem_alloc = 0;
  long mem_delete = 2*sizeof(double)*M*N;
  long start = get_time();
#pragma omp target teams distribute parallel for map(to: Anew[0:M][0:N]) \
                   map(from: A[0:M][0:N])
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      A[i][j] = Anew[i][j];      
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel2_gpu_mem,%ld,1,1,%ld,%ld,%ld,%ld,2,%d,%d\n", 
          (end - start), mem_to, mem_alloc, mem_from, mem_delete, M, N);
}
