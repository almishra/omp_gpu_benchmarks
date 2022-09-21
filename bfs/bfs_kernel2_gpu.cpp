#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 2 - GPU
////////////////////////////////////////////////////////////////////////////////
bool kernel2_gpu(bool *graph_mask, bool *updating_graph_mask,
                 bool *graph_visited, bool stop, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp target teams distribute parallel for map(tofrom: stop) \
                                                 map(num_threads, num_teams)
  for(int tid = 0; tid < N ; tid++) {
    if(tid == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
    if (updating_graph_mask[tid] == true) {
      graph_mask[tid] = true;
      graph_visited[tid] = true;
      stop = true;
      updating_graph_mask[tid] = false;
    }
  }
  long end = get_time();
  fprintf(fp, "bfs_kernel2_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,1,%d\n",
              (end - start), num_teams, num_threads, 2*sizeof(int),
              2*sizeof(int), N);

  return stop;
}
