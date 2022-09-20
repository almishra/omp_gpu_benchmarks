#ifndef __CC_H__
#define __CC_H__

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <math.h>

#ifndef N
#define N 1000
#endif

long get_time();
void init(double *X);
 
/* Kernels for calculating the Pearson's correlation coefficient*/
double cc_kernel_cpu(double *X, double *Y, FILE *fp);
double cc_kernel_gpu(double *X, double *Y, FILE *fp);
double cc_kernel_gpu_mem(double *X, double *Y, FILE *fp);

#endif
