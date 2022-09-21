#include "laplace.h"

void kernel2_gpu_collapse_mem(double (*A)[N], double (*Anew)[N], FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
  long mem_to = sizeof(double)*M*N + 2*sizeof(int);
  long mem_from = sizeof(double)*M*N + 2*sizeof(int);
  long mem_alloc = sizeof(double)*M*N;
  long mem_delete = sizeof(double)*M*N;

  long start = get_time();
#pragma omp target teams distribute parallel for collapse(2) \
                   map(to: Anew[0:M][0:N]) map(from: A[0:M][0:N]) map(num_teams, num_threads)
  for(int i = 1; i < M-1; i++) {
    for(int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
      A[i][j] = Anew[i][j];
    }
  }
  long end = get_time();

  fprintf(fp, "laplace_kernel2_gpu_collapse_mem,%ld,1,2,%d,%d,%ld,%ld,%ld,%ld,2,%d,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_delete, M, N);
}
