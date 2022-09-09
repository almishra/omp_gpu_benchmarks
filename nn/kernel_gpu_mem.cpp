#include "nn.h"

void kernel_nn_gpu_mem(float *z, float *lat, float *lon, FILE *fp) {
  long mem_to = 2*sizeof(float)*REC_WINDOW;
  long mem_from = sizeof(float)*REC_WINDOW;
  long mem_alloc = sizeof(float)*REC_WINDOW;
  long mem_del = 2*sizeof(float)*REC_WINDOW;

  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(alloc: z[0:REC_WINDOW]) \
                   map(to: lat[0:REC_WINDOW], lon[0:REC_WINDOW]) \
                   map(from: z[0:REC_WINDOW])
  for (int i = 0; i < REC_WINDOW; i++) {
    z[i] = (lat[i] - TARGET_LAT) * (lat[i] - TARGET_LAT) +
           (lon[i] - TARGET_LON) * (lon[i] - TARGET_LON);
  }
  long end = get_time();

  fprintf(fp, "kernel_nn_gpu_mem,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc,
          mem_from, mem_del, REC_WINDOW, end - start);
}
