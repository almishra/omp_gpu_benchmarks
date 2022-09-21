# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp -DDEBUG
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
N1=10000000

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: covariance.cpp covariance_kernel1_cpu.cpp covariance_kernel1_gpu.cpp covariance_kernel1_gpu_mem.cpp covariance_kernel2_cpu.cpp covariance_kernel2_gpu.cpp covariance_kernel2_gpu_mem.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance.cpp -o covariance_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel1_cpu.cpp -o covariance_kernel1_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel1_gpu.cpp -o covariance_kernel1_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel1_gpu_mem.cpp -o covariance_kernel1_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel2_cpu.cpp -o covariance_kernel2_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel2_gpu.cpp -o covariance_kernel2_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c covariance_kernel2_gpu_mem.cpp -o covariance_kernel2_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) covariance_kernel?_*_test.o covariance_test.o -o covariance_test.out
	$(RM) covariance*_test.o

clean:
	$(RM) *.o *.out
