# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp -DDEBUG
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
N1=10
N2=5

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: transpose.cpp transpose_kernel_cpu.cpp transpose_kernel_gpu.cpp transpose_kernel_gpu_mem.cpp transpose_kernel_cpu_collapse.cpp transpose_kernel_gpu_collapse.cpp transpose_kernel_gpu_collapse_mem.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose.cpp -o transpose_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_cpu.cpp -o transpose_kernel_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_gpu.cpp -o transpose_kernel_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_gpu_mem.cpp -o transpose_kernel_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_cpu_collapse.cpp -o transpose_kernel_cpu_collapse_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_gpu_collapse.cpp -o transpose_kernel_gpu_collapse_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c transpose_kernel_gpu_collapse_mem.cpp -o transpose_kernel_gpu_collapse_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) transpose_kernel_*_test.o transpose_test.o -o transpose_test.out
	$(RM) transpose*_test.o

clean:
	$(RM) *.o *.out
