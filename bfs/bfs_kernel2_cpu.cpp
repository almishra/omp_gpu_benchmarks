#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 2 - CPU
////////////////////////////////////////////////////////////////////////////////
bool kernel2_cpu(bool *graph_mask, bool *updating_graph_mask,
                 bool *graph_visited, bool stop, FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp parallel for
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
  fprintf(fp, "bfs_kernel2_cpu,%ld,0,1,%d,%d,0,0,0,0,1,%d\n",
              (end - start), num_teams, num_threads, N);

  return stop;
}
