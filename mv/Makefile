# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
N1=100000
N2=10000

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: mv.cpp mv_kernel_cpu.cpp mv_kernel_gpu.cpp mv_kernel_gpu_mem.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c mv.cpp -o mv_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c mv_kernel_cpu.cpp -o mv_kernel_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c mv_kernel_gpu.cpp -o mv_kernel_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) -c mv_kernel_gpu_mem.cpp -o mv_kernel_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=$(N1) -DN2=$(N2) mv_kernel_*_test.o  mv_test.o -o mv_test.out
	$(RM) mv_kernel_*_test.o

clean:
	$(RM) *.o *.out
