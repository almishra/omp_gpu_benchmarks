# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test:
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace.cpp -o laplace_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_cpu.cpp -o laplace_kernel1_cpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_cpu_collapse.cpp -o laplace_kernel1_cpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_cpu.cpp -o laplace_kernel2_cpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_cpu_collapse.cpp -o laplace_kernel2_cpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_gpu_mem.cpp -o laplace_kernel1_gpu_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_gpu_collapse_mem.cpp -o laplace_kernel1_gpu_collapse_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_gpu_mem.cpp -o laplace_kernel2_gpu_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_gpu_collapse_mem.cpp -o laplace_kernel2_gpu_collapse_mem_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_gpu.cpp -o laplace_kernel1_gpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel1_gpu_collapse.cpp -o laplace_kernel1_gpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_gpu.cpp -o laplace_kernel2_gpu_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 -c laplace_kernel2_gpu_collapse.cpp -o laplace_kernel2_gpu_collapse_test.o; \
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DM=100 -DN=100 laplace_kernel1_cpu_test.o laplace_kernel1_cpu_collapse_test.o laplace_kernel2_cpu_test.o laplace_kernel2_cpu_collapse_test.o laplace_kernel1_gpu_mem_test.o laplace_kernel1_gpu_collapse_mem_test.o laplace_kernel2_gpu_mem_test.o laplace_kernel2_gpu_collapse_mem_test.o laplace_kernel1_gpu_test.o laplace_kernel1_gpu_collapse_test.o laplace_kernel2_gpu_test.o laplace_kernel2_gpu_collapse_test.o laplace_test.o -o laplace_test.out; \
	$(RM) laplace_kernel1_cpu_test.o laplace_kernel1_cpu_collapse_test.o laplace_kernel2_cpu_test.o laplace_kernel2_cpu_collapse_test.o laplace_kernel1_gpu_mem_test.o laplace_kernel1_gpu_collapse_mem_test.o laplace_kernel2_gpu_mem_test.o laplace_kernel2_gpu_collapse_mem_test.o laplace_kernel1_gpu_test.o laplace_kernel1_gpu_collapse_test.o laplace_kernel2_gpu_test.o laplace_kernel2_gpu_collapse_test.o laplace_test.o;

clean:
	$(RM) *.o *.out
