#ifndef __MM_HEADER__
#define __MM_HEADER__

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

long get_time();

void mm_kernel_cpu(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void mm_kernel_cpu_collapse(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void mm_kernel_gpu(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void mm_kernel_gpu_collapse(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void mm_kernel_gpu_mem(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void mm_kernel_gpu_collapse_mem(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);

#endif
