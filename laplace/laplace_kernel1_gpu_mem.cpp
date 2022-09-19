#include "laplace.h"

double kernel1_gpu_mem(double (*A)[N], double (*Anew)[N], double err, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for (int i = 1; i < M-1; i++) {
    for (int j = 1; j < N-1; j++) {
      if(i == 1 && j == 1) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
      }
    }
  }
  long mem_to = sizeof(double)*M*N;
  long mem_from = sizeof(double)*M*N;
  long mem_alloc = sizeof(double)*M*N;
  long mem_delete = 2*sizeof(double)*M*N;
  long start = get_time();
#pragma omp target teams distribute parallel for reduction(max: err) \
                   map(alloc: Anew[0:M][0:N]) map(to: A[0:M][0:N]) map(err)
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

  fprintf(fp, "laplace_kernel1_gpu_mem,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,2,%d,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_delete, M, N);
  return err;
}
