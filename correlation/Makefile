# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp -DDEBUG
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
N1=10000000

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: cc.cpp cc_kernel_cpu.cpp cc_kernel_gpu.cpp cc_kernel_gpu_mem.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c cc.cpp -o cc_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c cc_kernel_cpu.cpp -o cc_kernel_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c cc_kernel_gpu.cpp -o cc_kernel_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -c cc_kernel_gpu_mem.cpp -o cc_kernel_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=$(N1) -lm cc_kernel_*_test.o cc_test.o -o cc_test.out
	$(RM) cc*_test.o

clean:
	$(RM) *.o *.out
