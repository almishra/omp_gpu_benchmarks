#include "mm.h"

long get_time()
{
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  return (long)(tv.tv_sec * 1000000 + tv.tv_usec);
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
  FILE *fp = fopen(output_file_name.c_str(), "w");

  double (*A)[N2] = (double (*)[N2]) malloc(sizeof(double)*N1*N2);
  double (*B)[N3] = (double (*)[N3]) malloc(sizeof(double)*N2*N3);
  double (*C)[N3] = (double (*)[N3]) malloc(sizeof(double)*N1*N3);

  // Initialize GPUs and check available memory
#pragma omp target enter data map(alloc: A[0:N1][0:N2], B[0:N2][0:N3], \
                                         C[0:N1][0:N3])
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3], \
                                         C[0:N1][0:N3])

  // CPU
  mm_kernel_cpu(A, B, C, fp);
  mm_kernel_cpu_collapse(A, B, C, fp);

  // GPU MEMCPY
  mm_kernel_gpu_mem(A, B, C, fp);
  mm_kernel_gpu_collapse_mem(A, B, C, fp);

  // GPU shared memory
#pragma omp target enter data map(to: A[0:N1][0:N2], B[0:N2][0:N3]) \
                              map(alloc: C[0:N1][0:N3])
  mm_kernel_gpu(A, B, C, fp);
  mm_kernel_gpu_collapse(A, B, C, fp);
#pragma omp target exit data map(delete: A[0:N1][0:N2], B[0:N2][0:N3]) \
    map(from: C[0:N1][0:N3])

  return 0;
}
