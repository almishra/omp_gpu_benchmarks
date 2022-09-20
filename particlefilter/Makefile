# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70
N = 1000000

all:
	@echo "Please run make for individual archs"
	@echo "make -f Makefile.<arch>"

test:
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter.cpp -o particlefilter_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel1_cpu.cpp -o particlefilter_kernel1_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel2_cpu.cpp -o particlefilter_kernel2_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel3_cpu.cpp -o particlefilter_kernel3_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel4_cpu.cpp -o particlefilter_kernel4_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel5_cpu.cpp -o particlefilter_kernel5_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel6_cpu.cpp -o particlefilter_kernel6_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel7_cpu.cpp -o particlefilter_kernel7_cpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel1_gpu.cpp -o particlefilter_kernel1_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel2_gpu.cpp -o particlefilter_kernel2_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel3_gpu.cpp -o particlefilter_kernel3_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel4_gpu.cpp -o particlefilter_kernel4_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel5_gpu.cpp -o particlefilter_kernel5_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel6_gpu.cpp -o particlefilter_kernel6_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel7_gpu.cpp -o particlefilter_kernel7_gpu_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel1_gpu_mem.cpp -o particlefilter_kernel1_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel2_gpu_mem.cpp -o particlefilter_kernel2_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel3_gpu_mem.cpp -o particlefilter_kernel3_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel4_gpu_mem.cpp -o particlefilter_kernel4_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel5_gpu_mem.cpp -o particlefilter_kernel5_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel6_gpu_mem.cpp -o particlefilter_kernel6_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -c -DN=$(N1) particlefilter_kernel7_gpu_mem.cpp -o particlefilter_kernel7_gpu_mem_test.o
	$(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -lm -DN=$(N) particlefilter_kernel?_cpu_test.o particlefilter_kernel?_gpu_test.o particlefilter_kernel?_gpu_mem_test.o particlefilter_test.o -o particlefilter_test.out
	$(RM) particlefilter_kernel?_cpu_test.o particlefilter_kernel?_gpu_test.o particlefilter_kernel?_gpu_mem_test.o particlefilter_test.o

clean:
	$(RM) *.o *.out
