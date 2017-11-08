CXX = icc #g++
MPICC = mpic++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 #-march=haswell
ADDITIONAL_FLAG = -D_DEBUG   #####debug purpose...-D_HOGWILD_SHORT 
#ADDITIONAL_FLAG = -D_PRINT_GRADIENT
#ADDITIONAL_FLAG = -DLOAD_FILE_PER_WORKER -D_DEBUG
CPP_INCLUDE = -I../modelsync_bitweaving/src -I../modelsync_bitweaving/hazytl/include/ -I../IntelPCM/
CPP_LAST = -lpthread

.PHONY: depend clean all
all: TEST_AVX TEST_MMAP 

clean: 
	rm -f TEST_AVX TEST_MMAP


TEST_AVX: test_fvector.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@


TEST_MMAP: test_mmap.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

