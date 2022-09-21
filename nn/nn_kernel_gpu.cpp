#include "nn.h"

void kernel_nn_gpu(float *z, float *lat, float *lon, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp target teams distribute parallel for map(num_teams, num_threads)
  for (int i = 0; i < REC_WINDOW; i++) {
    if(i == 0) {
        num_threads = omp_get_num_threads();
        num_teams = omp_get_num_teams();
    }
    z[i] = (lat[i] - TARGET_LAT) * (lat[i] - TARGET_LAT) +
           (lon[i] - TARGET_LON) * (lon[i] - TARGET_LON);
  }
  long end = get_time();

  fprintf(fp, "nn_kernel_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,1,%d\n",
          (end - start), num_teams, num_threads, 2*sizeof(int), 2*sizeof(int),
          REC_WINDOW);
}
