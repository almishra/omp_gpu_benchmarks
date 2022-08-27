#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <string> 

#ifndef N1
#define N1 1000
#endif

#ifndef N2
#define N2 1000
#endif

#ifndef N3
#define N3 1000
#endif

long mem_to;                                                                    
long mem_from;                                                                  
long mem_alloc;
long mem_del;
static FILE *fp;

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
}

void multiply_collapse1(double (*A)[N2], double (*B)[N3], double (*C)[N3])
{
  long start = get_time();
#pragma omp parallel for collapse(1)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "matrix_mult_col1,1,0,0,%ld,%ld,%d,%d,%d,%ld\n", mem_from, mem_del, N1, N2, N3, end - start);
}

void multiply_collapse2(double (*A)[N2], double (*B)[N3], double (*C)[N3])
{
  long start = get_time();
#pragma omp parallel for collapse(2)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
  long end = get_time();
  fprintf(fp, "matrix_mult_col2,2,0,0,%ld,%ld,%d,%d,%d,%ld\n", mem_from, mem_del, N1, N2, N3, end - start);
}

void multiply_gpu_collapse1(double (*A)[N2], double (*B)[N3], double (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(double)*(N1*N2 + N2*N3);
  mem_alloc = sizeof(double)*(N1*N3);
  mem_from = sizeof(double)*(N1*N3);
  mem_del = sizeof(double)*(N1*N2 + N2*N3);
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_gpu_col1,1,%ld,%ld,%ld,%ld,%d,%d,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N1, N2, N3, end - start);
}

void multiply_gpu_collapse2(double (*A)[N2], double (*B)[N3], double (*C)[N3])
{
#ifdef MEMCPY
  mem_to = sizeof(double)*(N1*N2 + N2*N3);
  mem_alloc = sizeof(double)*(N1*N3);
  mem_from = sizeof(double)*(N1*N3);
  mem_del = sizeof(double)*(N1*N2 + N2*N3);
#endif
  long start = get_time();
#ifdef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(alloc: C[0:N1][0:N3])
#endif
#pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i<N1; i++) {
    for(int j=0; j<N3; j++) {
      double sum = 0.0;
      for (int k = 0; k < N2; k++)
        sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }
#ifdef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(from: C[0:N1][0:N3])
#endif
  long end = get_time();
  fprintf(fp, "matrix_mult_gpu_col2,2,%ld,%ld,%ld,%ld,%d,%d,%d,%ld\n", mem_to, mem_alloc, mem_from, mem_del, N1, N2, N3, end - start);
}

int main(int argc, char **argv)
{
  std::string output_file_name;
  if(argc > 1) {
    output_file_name = argv[1];
  } else {
    output_file_name = argv[0];
    output_file_name = output_file_name.substr(output_file_name.find_last_of("/\\")+1);
    output_file_name = output_file_name.substr(0, output_file_name.size() - 3);
    output_file_name = "output_" + output_file_name + "csv";
  }

  printf("%s\n", output_file_name.c_str());
  fp = fopen(output_file_name.c_str(), "w");

  fprintf(fp, "Total size for double = %0.4lfGB\n", sizeof(double)*(N1*N2 + N2*N3 + N1*N3)/1024.0/1024.0/1024.0);

  double (*A)[N2] = (double (*)[N2]) malloc(sizeof(double)*N1*N2);
  double (*B)[N3] = (double (*)[N3]) malloc(sizeof(double)*N2*N3);
  double (*C)[N3] = (double (*)[N3]) malloc(sizeof(double)*N1*N3);

#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
  map(alloc: C[0:N1][0:N3])
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(from: C[0:N1][0:N3])

#ifndef MEMCPY
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(alloc: C[0:N1][0:N3])
  multiply_collapse1(A, B, C);
  multiply_collapse2(A, B, C);
#endif
  multiply_gpu_collapse1(A, B, C);
  multiply_gpu_collapse2(A, B, C);
#ifndef MEMCPY
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(from: C[0:N1][0:N3])
#endif
  return 0;
}
