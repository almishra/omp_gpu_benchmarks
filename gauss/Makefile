# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test:
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss.cpp -o gauss_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_cpu.cpp -o gauss_kernel_cpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_cpu_collapse.cpp -o gauss_kernel_cpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_gpu.cpp -o gauss_kernel_gpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_gpu_collapse.cpp -o gauss_kernel_gpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_gpu_mem.cpp -o gauss_kernel_gpu_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c gauss_kernel_gpu_collapse_mem.cpp -o gauss_kernel_gpu_collapse_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 gauss_kernel_cpu_test.o gauss_kernel_cpu_collapse_test.o gauss_kernel_gpu_test.o gauss_kernel_gpu_collapse_test.o gauss_kernel_gpu_mem_test.o gauss_kernel_gpu_collapse_mem_test.o gauss_test.o -o gauss_test.out; \
	$(RM) gauss_kernel_cpu_test.o gauss_kernel_cpu_collapse_test.o gauss_kernel_gpu_test.o gauss_kernel_gpu_collapse_test.o gauss_kernel_gpu_mem_test.o gauss_kernel_gpu_collapse_mem_test.o gauss_test.o; \

clean:
	$(RM) *.o *.out
