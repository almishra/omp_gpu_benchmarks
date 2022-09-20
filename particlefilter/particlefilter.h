#ifndef __PARTICLEFILTER_H__
#define __PARTICLEFILTER_H__
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>
#include <string>
#include <ctime>

#define PI 3.1415926535897932
#define M INT_MAX
#define A 1103515245
#define C 12345

#ifndef N
#define N 1000
#endif

long get_time();
//float elapsed_time(long long start_time, long long end_time);
double roundDouble(double value);
void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ);
double randu(int * seed, int index);
double randn(int * seed, int index);
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed);
void strelDisk(int * disk, int radius);
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error);
void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix);
void getneighbors(int * se, int numOnes, double * neighbors, int radius);
void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed);

void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, FILE *fp);

void particlefilter_kernel1_cpu(double *weights, double *arrayX, double *arrayY, double xe, double ye, FILE *fp);
double particlefilter_kernel2_cpu(double *weights, FILE *fp);
void particlefilter_kernel3_cpu(double *weights, double sumWeights, FILE *fp);
void particlefilter_kernel4_cpu(double *arrayX, double *arrayY, double *weights, double &xe, double &ye, FILE *fp);
void particlefilter_kernel5_cpu(double *u, double ux, FILE *fp);
void particlefilter_kernel6_cpu(double *CDF, double *u, double *arrayX, double *arrayY, double *xj, double *xy, FILE *fp);
void particlefilter_kernel7_cpu(double *weights, double *arrayX, double *arrayY, double *xj, double *yj, FILE *fp);

void particlefilter_kernel1_gpu(double *weights, double *arrayX, double *arrayY, double xe, double ye, FILE *fp);
double particlefilter_kernel2_gpu(double *weights, FILE *fp);
void particlefilter_kernel3_gpu(double *weights, double sumWeights, FILE *fp);
void particlefilter_kernel4_gpu(double *arrayX, double *arrayY, double *weights, double &xe, double &ye, FILE *fp);
void particlefilter_kernel5_gpu(double *u, double ux, FILE *fp);
void particlefilter_kernel6_gpu(double *CDF, double *u, double *arrayX, double *arrayY, double *xj, double *xy, FILE *fp);
void particlefilter_kernel7_gpu(double *weights, double *arrayX, double *arrayY, double *xj, double *yj, FILE *fp);

void particlefilter_kernel1_gpu_mem(double *weights, double *arrayX, double *arrayY, double xe, double ye, FILE *fp);
double particlefilter_kernel2_gpu_mem(double *weights, FILE *fp);
void particlefilter_kernel3_gpu_mem(double *weights, double sumWeights, FILE *fp);
void particlefilter_kernel4_gpu_mem(double *arrayX, double *arrayY, double *weights, double &xe, double &ye, FILE *fp);
void particlefilter_kernel5_gpu_mem(double *u, double ux, FILE *fp);
void particlefilter_kernel6_gpu_mem(double *CDF, double *u, double *arrayX, double *arrayY, double *xj, double *xy, FILE *fp);
void particlefilter_kernel7_gpu_mem(double *weights, double *arrayX, double *arrayY, double *xj, double *yj, FILE *fp);

#endif
