#ifndef __COVARIANCE_H__
#define __COVARIANCE_H__

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
 
/* Kernels for calculating the Covariance */
double covariance_kernel1_cpu(double *X, FILE *fp);
double covariance_kernel1_gpu(double *X, FILE *fp);
double covariance_kernel1_gpu_mem(double *X, FILE *fp);
double covariance_kernel2_cpu(double *X, double *Y, double meanX, double meanY, FILE *fp);
double covariance_kernel2_gpu(double *X, double *Y, double meanX, double meanY, FILE *fp);
double covariance_kernel2_gpu_mem(double *X, double *Y, double meanX, double meanY, FILE *fp);

#endif
