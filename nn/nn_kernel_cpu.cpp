#include "nn.h"

void kernel_nn_cpu(float *z, float *lat, float *lon, FILE *fp) {
  long start = get_time();
#pragma omp parallel for
  for (int i = 0; i < REC_WINDOW; i++) {
    z[i] = (lat[i] - TARGET_LAT) * (lat[i] - TARGET_LAT) +
           (lon[i] - TARGET_LON) * (lon[i] - TARGET_LON);
  }
  long end = get_time();

  fprintf(fp, "nn_kernel_cpu,%ld,0,1,0,0,0,0,1,%d\n", (end - start), REC_WINDOW);
}
