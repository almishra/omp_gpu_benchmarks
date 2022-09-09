# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test:
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs.cpp -o bfs_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel1_cpu.cpp -o bfs_kernel1_cpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel2_cpu.cpp -o bfs_kernel2_cpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel1_gpu.cpp -o bfs_kernel1_gpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel2_gpu.cpp -o bfs_kernel2_gpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel1_gpu_mem.cpp -o bfs_kernel1_gpu_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 -c bfs_kernel2_gpu_mem.cpp -o bfs_kernel2_gpu_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DN=100 bfs_kernel1_cpu_test.o bfs_kernel2_cpu_test.o bfs_kernel1_gpu_test.o bfs_kernel2_gpu_test.o bfs_kernel1_gpu_mem_test.o bfs_kernel2_gpu_mem_test.o bfs_test.o -o bfs_test.out; \
	$(RM) bfs_kernel1_cpu_test.o bfs_kernel2_cpu_test.o bfs_kernel1_gpu_test.o bfs_kernel2_gpu_test.o bfs_kernel1_gpu_mem_test.o bfs_kernel2_gpu_mem_test.o bfs_test.o;

clean:
	$(RM) *.o *.out
