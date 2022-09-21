#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 1 - GPU
////////////////////////////////////////////////////////////////////////////////
void kernel1_gpu(Node* graph_nodes, bool *graph_mask, bool *updating_graph_mask,
             bool *graph_visited, int *graph_edges, int *cost, int totalEdges,
             FILE *fp)
{
  int num_threads = 0;
  int num_teams = 1;

  long start = get_time();
#pragma omp target teams distribute parallel for map(num_threads, num_teams)
  for(int tid = 0; tid < N; tid++ ) {
    if(tid == 0) {
      num_threads = omp_get_num_threads();
      num_teams = omp_get_num_teams();
    }
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
  fprintf(fp, "bfs_kernel1_gpu,%ld,1,1,%d,%d,%lu,0,%lu,0,1,%d\n",
              (end - start), num_teams, num_threads, 2*sizeof(int),
              2*sizeof(int), N);
}
