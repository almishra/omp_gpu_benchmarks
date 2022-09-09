#ifndef __LAPLACE_H__
#define __LAPLACE_H__
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>

#ifndef M
#define M 100
#endif

#ifndef N
#define N 100
#endif

#define MAX_ITER 20
#define TOL 0.000001

void initialize(double alpha, double (*A)[N], double (*A1)[N], double (*A2)[N], 
                double (*A3)[N], double (*A4)[N], double (*A5)[N]);
long get_time();
double kernel1_cpu(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
double kernel1_cpu_collapse(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
double kernel1_gpu_mem(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
double kernel1_gpu_collapse_mem(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
double kernel1_gpu(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
double kernel1_gpu_collapse(double (*A)[N], double (*Anew)[N], double err, FILE *fp);
void kernel2_cpu(double (*A)[N], double (*Anew)[N], FILE *fp);
void kernel2_cpu_collapse(double (*A)[N], double (*Anew)[N], FILE *fp);
void kernel2_gpu_mem(double (*A)[N], double (*Anew)[N], FILE *fp);
void kernel2_gpu_collapse_mem(double (*A)[N], double (*Anew)[N], FILE *fp);
void kernel2_gpu(double (*A)[N], double (*Anew)[N], FILE *fp);
void kernel2_gpu_collapse(double (*A)[N], double (*Anew)[N], FILE *fp);
#endif
