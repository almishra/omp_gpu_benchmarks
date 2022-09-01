#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 2 - GPU MEMCPY
////////////////////////////////////////////////////////////////////////////////
bool kernel2_gpu_mem(bool *graph_mask, bool *updating_graph_mask, bool *graph_visited,
             bool stop, FILE *fp)
{
  long mem_to = 0;
  long mem_from = 0;
  long mem_alloc = 0;
  long mem_del = 0;

  mem_to = sizeof(bool)*N + sizeof(bool)*N + sizeof(bool)*N + sizeof(int);
  mem_from = mem_to;
  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(updating_graph_mask[0:N], graph_mask[0:N], \
                       graph_visited[0:N], stop)
  for(int tid = 0; tid < N ; tid++) {
    if (updating_graph_mask[tid] == true) {
      graph_mask[tid] = true;
      graph_visited[tid] = true;
      stop = true;
      updating_graph_mask[tid] = false;
    }
  }
  long end = get_time();

  fprintf(fp, "bfs_kernel2_gpu_mem,%ld,%ld,%ld,%ld,%d,%ld\n",
          mem_to, mem_alloc, mem_from, mem_del, N, (end - start));

  return stop;
}
