CXX = icc #g++
MPICC = mpic++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 #-march=haswell
ADDITIONAL_FLAG = -D_DEBUG   #####debug purpose...-D_HOGWILD_SHORT 
#ADDITIONAL_FLAG = -D_PRINT_GRADIENT
#ADDITIONAL_FLAG = -DLOAD_FILE_PER_WORKER -D_DEBUG
CPP_INCLUDE = -I../modelsync_bitweaving/src -I../modelsync_bitweaving/hazytl/include/ -I../IntelPCM/  -I../tool/
CPP_LAST = -lpthread

COMMON_O   = rand_tool.o file_tool.o cpu_mapping.o huge_page.o

COMMON_CPP = rand_tool.cpp file_tool.cpp cpu_mapping.cpp huge_page.cpp


.PHONY: depend clean all
all: TEST_AVX TEST_MMAP  #$(COMMON_O)

clean: 
	rm -f TEST_AVX TEST_MMAP


TEST_AVX: test_fvector.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@


TEST_MMAP: test_mmap.cpp
	#mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@

${TEST_AVX2}: test/test_fvector.cpp
	mkdir -p $(BIN_DIR)
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} $< -o $@





#####################common functions##########################################
rand_tool.o: rand_tool.cpp   
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c rand_tool.cpp

thread_tool.o: thread_tool.cpp   
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c thread_tool.cpp

file_tool.o: file_tool.cpp   
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c file_tool.cpp

huge_page.o: huge_page.cpp 
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c huge_page.cpp

cpu_mapping.o: cpu_mapping.cpp 
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c cpu_mapping.cpp

memory_tool.o: memory_tool.cpp   
	$(CXX)  ${CPP_FLAG} ${CPP_INCLUDE} -c memory_tool.cpp
#########################end of common functions################################
