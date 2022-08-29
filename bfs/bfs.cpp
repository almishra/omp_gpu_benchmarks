#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <vector>

#ifndef N                                                                      
#define N 1000                                                                 
#endif

int source;
long totalEdges;

long get_time()                                                                 
{                                                                               
  struct timeval  tv;                                                           
  gettimeofday(&tv, NULL);                                                      
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);                              
} 

long mem_to = 0;
long mem_from = 0;
long mem_alloc = 0;
long mem_del = 0;

//Structure to hold a node information
struct Node
{
  int start;
  int num_edges;
};

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  srand(0);
  BFSGraph(argc, argv);
}

struct edge;
typedef std::vector<edge> node;                                                      
struct edge {                                                                   
  ulong dest;                                                                   
  uint weight;                                                                  
};

#define MIN_NODES 20
#define MAX_NODES ULONG_MAX
#define MIN_EDGES 2
#define MAX_INIT_EDGES 4 // Nodes will have, on average, 2*MAX_INIT_EDGES edges
#define MIN_WEIGHT 1
#define MAX_WEIGHT 10

////////////////////////////////////////////////////////////////////////////////
// Create Graph Funtion
// Returns Graph edges
////////////////////////////////////////////////////////////////////////////////
int* CreateGraph(Node* graph_nodes, bool *graph_mask,
    bool *updating_graph_mask, bool *graph_visited) {
  int source = 0;

  node * graph = new node[N];
  uint numEdges;
  ulong nodeID;
  uint weight;
  uint j;

  for (long i = 0; i < N; i++ ) {
    numEdges = rand() % ( MAX_INIT_EDGES - MIN_EDGES + 1 ) + MIN_EDGES;
    for ( j = 0; j < numEdges; j++ ) {
      //      nodeID = randNode( gen );
      nodeID = rand() % numEdges;
      weight = rand() % ( MAX_WEIGHT - MIN_WEIGHT + 1 ) + MIN_WEIGHT;
      graph[i].push_back( edge() );
      graph[i].back().dest = nodeID;
      graph[i].back().weight = weight;
      graph[nodeID].push_back( edge() );
      graph[nodeID].back().dest = i;
      graph[nodeID].back().weight = weight;
    }
  }

  int start, edgeno;
  totalEdges = 0;
  for (long i = 0; i < N; i++ ) {
    numEdges = graph[i].size();
    start = totalEdges;
    edgeno = numEdges;
    totalEdges += numEdges;
    graph_nodes[i].start = start;
    graph_nodes[i].num_edges = edgeno;
    graph_mask[i]=false;
    updating_graph_mask[i]=false;
    graph_visited[i]=false;
  }

  source = rand() % numEdges;
  graph_mask[source] = true;
  graph_visited[source] = true;

  int* graph_edges = (int*) malloc(sizeof(int) * totalEdges);

  int k = 0;
  for (long i = 0; i < N; i++ ) {
    for ( uint j = 0; j < graph[i].size(); j++ ) {
      graph_edges[k++] = graph[i][j].dest;
    }
  }

  return graph_edges;
}

////////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph using OpenMP
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
  // allocate memory
  Node *graph_nodes = (Node*) malloc(sizeof(Node) * N);
  bool *graph_mask = (bool*) malloc(sizeof(bool) * N);
  bool *updating_graph_mask = (bool*) malloc(sizeof(bool) * N);
  bool *graph_visited = (bool*) malloc(sizeof(bool) * N);
  int  *graph_edges = CreateGraph(graph_nodes, graph_mask, updating_graph_mask, 
                                  graph_visited);

  // allocate mem for the result 
  int* cost = (int*) malloc( sizeof(int)*N);
  for(int i=0;i<N;i++)
    cost[i]=-1;
  cost[source]=0;

  // Initiate GPUs and check if it has enough memory
#ifdef OMP_OFFLOAD
#pragma omp target enter data map(to: graph_mask[0:N], \
                                      graph_nodes[0:N], \
                                      graph_edges[0:totalEdges], \
                                      graph_visited[0:N], \
                                      updating_graph_mask[0:N], \
                                      cost[0:N])
#pragma omp target exit data map(delete: graph_mask[0:N], \
                                      graph_nodes[0:N], \
                                      graph_edges[0:totalEdges], \
                                      graph_visited[0:N], \
                                      updating_graph_mask[0:N], \
                                      cost[0:N])
#endif


#ifdef OMP_OFFLOAD
#ifndef OMP_OFFLOAD_MEMCPY
  mem_to += sizeof(bool)*N + sizeof(Node)*N + sizeof(int)*totalEdges +
            sizeof(bool)*N + sizeof(bool)*N + sizeof(int)*N;
  mem_from += sizeof(int)*N;
#pragma omp target enter data map(to: graph_mask[0:N], \
                                      graph_nodes[0:N], \
                                      graph_edges[0:totalEdges], \
                                      graph_visited[0:N], \
                                      updating_graph_mask[0:N], \
                                      cost[0:N])
#endif
  {
#endif 
    bool stop = true;
    while(stop) {
      // if no thread changes this value then the loop stops
      stop = false;
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
      mem_to = sizeof(Node)*N + sizeof(int)*totalEdges + sizeof(bool)*N + 
                sizeof(int)*N + sizeof(bool)*N + sizeof(bool)*N;
#endif
#endif
      long start = get_time();
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
#pragma omp target enter data map(to: graph_nodes[0:N], \
                                      graph_edges[0:totalEdges], \
                                      graph_visited[0:N], \
                                      cost[0:N], \
                                      updating_graph_mask[0:N], \
                                      graph_mask[0:N])
#endif
#pragma omp target teams distribute parallel for
#else
#pragma omp parallel for 
#endif
      for(int tid = 0; tid < N; tid++ ) {
        if (graph_mask[tid] == true) { 
          graph_mask[tid]=false;
          int num = graph_nodes[tid].num_edges + graph_nodes[tid].start;
          for(int i = graph_nodes[tid].start; i < num; i++) {
            int id = graph_edges[i];
            if(!graph_visited[id]) {
              cost[id]=cost[tid]+1;
              updating_graph_mask[id]=true;
            }
          }
        }
      }
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
#pragma omp target exit data map(from: graph_nodes[0:N], \
                                      graph_edges[0:totalEdges], \
                                      graph_visited[0:N], \
                                      cost[0:N], \
                                      updating_graph_mask[0:N], \
                                      graph_mask[0:N])
#endif
#endif
      long end = get_time();
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
      mem_from = sizeof(Node)*N + sizeof(int)*totalEdges + sizeof(bool)*N + sizeof(int)*N + sizeof(bool)*N + sizeof(bool)*N;
#endif
#endif
#ifdef OMP_OFFLOAD
      printf("bfs_kernel_gpu1,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N, (end - start));
#else
      printf("bfs_kernel_cpu1,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N, (end - start));
#endif

#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
      mem_to = sizeof(bool)*N + sizeof(bool)*N + sizeof(bool)*N;
#endif
      mem_to += sizeof(int); // Size of stop
#endif
      start = get_time();
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
#pragma omp target enter data map(to: updating_graph_mask[0:N], \
                                      graph_mask[0:N], \
                                      graph_visited[0:N])
#endif
#pragma omp target teams distribute parallel for map(stop)
#else
#pragma omp parallel for
#endif
      for(int tid = 0; tid < N ; tid++) {
        if (updating_graph_mask[tid] == true) {
          graph_mask[tid]=true;
          graph_visited[tid]=true;
          stop=true;
          updating_graph_mask[tid]=false;
        }
      }
      end = get_time();
#ifdef OMP_OFFLOAD
#ifdef OMP_OFFLOAD_MEMCPY
      mem_from = sizeof(bool)*N + sizeof(bool)*N + sizeof(bool)*N;
#endif
      mem_from += sizeof(int); // Size of stop
#endif

#ifdef OMP_OFFLOAD
      printf("bfs_kernel_gpu2,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N, (end - start));
#else
      printf("bfs_kernel_cpu2,%ld,%ld,%ld,%ld,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N, (end - start));
#endif
    }
#ifdef OMP_OFFLOAD
  }
#pragma omp target exit data map(delete: graph_mask[0:N], graph_nodes[0:N], graph_edges[0:totalEdges], graph_visited[0:N], updating_graph_mask[0:N]) map(from: cost[0:N])
#endif

#ifdef DEBUG
  //Store the result into a file
  FILE *fpo = fopen("result.txt","w");
  for(int i=0;i<N;i++) fprintf(fpo,"%d) cost:%d\n",i,cost[i]);
  fclose(fpo);
  printf("Result stored in result.txt\n");
#endif

  // cleanup memory
  free(graph_nodes);
  free(graph_edges);
  free(graph_mask);
  free(updating_graph_mask);
  free(graph_visited);
  free(cost);

}

