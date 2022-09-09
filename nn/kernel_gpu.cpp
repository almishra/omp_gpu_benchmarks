#include "nn.h"

void kernel_nn_gpu(float *z, float *lat, float *lon, FILE *fp) {
  long start = get_time();
#pragma omp target teams distribute parallel for
  for (int i = 0; i < REC_WINDOW; i++) {
    z[i] = (lat[i] - TARGET_LAT) * (lat[i] - TARGET_LAT) +
           (lon[i] - TARGET_LON) * (lon[i] - TARGET_LON);
  }
  long end = get_time();

  fprintf(fp, "kernel_nn_gpu,0,0,0,0,%d,%ld\n", REC_WINDOW, end - start);
}
