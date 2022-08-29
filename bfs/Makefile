# C++ compiler
CXX = clang++
CFLAGS = -Wall -O3 -fopenmp
OFFLOAD_FLAGS = -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_70

bfs: bfs.cpp
	for (( N=5000; N<=10000; N+=1000 )); do \
		$(CXX) $(CFLAGS) bfs.cpp -o bfs_$${N}.out -DN=$${N}; \
	  $(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload_$${N}.out -DN=$${N}; \
	  $(CXX) $(CFLAGS) $(OFFLOAD_FLAGS) -DOMP_OFFLOAD -DOMP_OFFLOAD_MEMCPY bfs.cpp -o bfs_offload_memcpy_$${N}.out -DN=$${N}; \
	done

clean:
	rm -f *.out
