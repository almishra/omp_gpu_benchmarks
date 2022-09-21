#include "nn.h"

void kernel_nn_gpu_mem(float *z, float *lat, float *lon, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;
  long mem_to = 2*sizeof(float)*REC_WINDOW + 2*sizeof(int);
  long mem_from = sizeof(float)*REC_WINDOW + 2*sizeof(int);
  long mem_alloc = sizeof(float)*REC_WINDOW;
  long mem_del = 2*sizeof(float)*REC_WINDOW;

  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(alloc: z[0:REC_WINDOW]) \
                   map(to: lat[0:REC_WINDOW], lon[0:REC_WINDOW]) \
                   map(from: z[0:REC_WINDOW]) map(num_teams, num_threads)
  for (int i = 0; i < REC_WINDOW; i++) {
    if(i == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
    }
    z[i] = (lat[i] - TARGET_LAT) * (lat[i] - TARGET_LAT) +
           (lon[i] - TARGET_LON) * (lon[i] - TARGET_LON);
  }
  long end = get_time();

  fprintf(fp, "nn_kernel_gpu_mem,%ld,1,1,%d,%d,%ld,%ld,%ld,%ld,1,%d\n",
          (end - start), num_teams, num_threads, mem_to, mem_alloc, mem_from,
          mem_del, REC_WINDOW);
}
