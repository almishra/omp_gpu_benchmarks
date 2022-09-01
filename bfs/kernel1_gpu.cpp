#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 1 - GPU
////////////////////////////////////////////////////////////////////////////////
void kernel1_gpu(Node* graph_nodes, bool *graph_mask, bool *updating_graph_mask,
             bool *graph_visited, int *graph_edges, int *cost, int totalEdges,
             FILE *fp)
{
  long start = get_time();
#pragma omp target teams distribute parallel for
  for(int tid = 0; tid < N; tid++ ) {
    if (graph_mask[tid] == true) { 
      graph_mask[tid] = false;
      int num = graph_nodes[tid].num_edges + graph_nodes[tid].start;
      for(int i = graph_nodes[tid].start; i < num; i++) {
        int id = graph_edges[i];
        if(!graph_visited[id]) {
          cost[id] = cost[tid] + 1;
          updating_graph_mask[id] = true;
        }
      }
    }
  }
  long end = get_time();
  fprintf(fp, "bfs_kernel1_gpu,0,0,0,0,%d,%ld\n", N, (end - start));
}
