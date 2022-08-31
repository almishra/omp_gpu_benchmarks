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

void multiply_collapse1(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void multiply_collapse2(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void multiply_gpu_collapse1(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);
void multiply_gpu_collapse2(double (*A)[N2], double (*B)[N3], double (*C)[N3], FILE *fp);

#endif
