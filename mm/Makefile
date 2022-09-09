# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: mm.cpp mm_kernel_cpu_collapse.cpp mm_kernel_cpu.cpp mm_kernel_gpu_collapse.cpp mm_kernel_gpu_collapse_mem.cpp mm_kernel_gpu.cpp mm_kernel_gpu_mem.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_cpu.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_cpu_collapse.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_cpu_collapse_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_gpu.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_gpu_collapse.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_gpu_collapse_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_gpu_mem.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 -c mm_kernel_gpu_collapse_mem.cpp -DN1=100 -DN2=100 -DN3=100 -o mm_kernel_gpu_collapse_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN1=100 -DN2=100 -DN3=100 mm_kernel_cpu_test.o mm_kernel_cpu_collapse_test.o mm_kernel_gpu_test.o mm_kernel_gpu_collapse_test.o mm_kernel_gpu_mem_test.o mm_kernel_gpu_collapse_mem_test.o mm_test.o -o mm_test.out
	$(RM) *_test.o

clean:
	$(RM) *.o *.out
