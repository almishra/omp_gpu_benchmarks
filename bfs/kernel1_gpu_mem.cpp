#include "bfs.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel 1 - GPU MEMCPY
////////////////////////////////////////////////////////////////////////////////
void kernel1_gpu_mem(Node* graph_nodes, bool *graph_mask,
                     bool *updating_graph_mask, bool *graph_visited,
                     int *graph_edges, int *cost, int totalEdges, FILE *fp)
{
  long mem_to = 0;
  long mem_from = 0;
  long mem_alloc = 0;
  long mem_del = 0;

  mem_to = sizeof(Node)*N + sizeof(int)*totalEdges + sizeof(bool)*N + 
           sizeof(int)*N + sizeof(bool)*N + sizeof(bool)*N;
  mem_from = mem_to;
 
  long start = get_time();
#pragma omp target teams distribute parallel for \
                   map(graph_nodes[0:N], graph_edges[0:totalEdges], \
                       graph_visited[0:N], cost[0:N], updating_graph_mask[0:N],\
                       graph_mask[0:N])
  for(int tid = 0; tid < N; tid++ ) {
    if (graph_mask[tid] == true) { 
      graph_mask[tid]=false;
      int num = graph_nodes[tid].num_edges + graph_nodes[tid].start;
      for(int i = graph_nodes[tid].start; i < num; i++) {
        int id = graph_edges[i];
        if(!graph_visited[id]) {
          cost[id] = cost[tid]+1;
          updating_graph_mask[id] = true;
        }
      }
    }
  }
  long end = get_time();

  fprintf(fp, "bfs_kernel1_gpu_mem,%ld,%ld,%ld,%ld,%d,%ld\n",
          mem_to, mem_alloc, mem_from, mem_del, N, (end - start));
}
