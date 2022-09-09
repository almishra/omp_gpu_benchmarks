#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 2 - CPU
////////////////////////////////////////////////////////////////////////////////
bool kernel2_cpu(bool *graph_mask, bool *updating_graph_mask,
                 bool *graph_visited, bool stop, FILE *fp)
{
  long start = get_time();
#pragma omp parallel for
  for(int tid = 0; tid < N ; tid++) {
    if (updating_graph_mask[tid] == true) {
      graph_mask[tid] = true;
      graph_visited[tid] = true;
      stop = true;
      updating_graph_mask[tid] = false;
    }
  }
  long end = get_time();
  fprintf(fp, "bfs_kernel2_cpu,%ld,0,1,0,0,0,0,1,%d\n", (end - start), N);

  return stop;
}
