CXX = icc #g++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 -lpthread -lm
CPP_INCLUDE = -I../tool/ -I../hazytl/include/ -I ../BitWeaving/
CPP_LAST = -lpthread -lm

COMMON_CPP = ../tool/rand_tool.cpp  ../tool/cpu_mapping.cpp ../tool/huge_page.cpp  ../tool/thread_tool.cpp

.PHONY: depend clean all
all: TEST_MMAP TEST_MEMORY_BANDWIDTH TEST_BITWEAVING #$(COMMON_O)

clean: 
	rm -f TEST_MMAP TEST_MEMORY_BANDWIDTH TEST_BITWEAVING


TEST_MMAP: test_mmap.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

TEST_MEMORY_BANDWIDTH: test_memory_bandwidth.cpp $(COMMON_CPP) # huge_page.o #
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} test_memory_bandwidth.cpp $(COMMON_CPP) -o $@ $(CPP_LAST)

TEST_BITWEAVING: test_bitweaving.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@


