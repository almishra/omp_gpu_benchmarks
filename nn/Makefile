# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test: nn.cpp
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c nn.cpp -o nn_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c kernel_cpu.cpp -o kernel_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c kernel_gpu.cpp -o kernel_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c kernel_gpu_mem.cpp -o kernel_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -lm kernel_cpu_test.o kernel_gpu_test.o kernel_gpu_mem_test.o nn_test.o -o nn_test.out
	$(RM) *_test.o

clean:
	$(RM) *.out *.o
