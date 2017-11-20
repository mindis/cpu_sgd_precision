CXX = icc #g++
CPP_FLAG = -O3 -std=c++11 -lrt #-Wall -fopenmp -Wno-unknown-pragmas -mavx2 -lpthread -lm
CPP_INCLUDE = -I../tool/ -I../hazytl/include/ -I ../BitWeaving/
CPP_LAST = -lpthread -lm

COMMON_CPP = ../tool/rand_tool.cpp  ../tool/cpu_mapping.cpp ../tool/huge_page.cpp  ../tool/thread_tool.cpp

.PHONY: depend clean all
all: WRITE_TO_BITWEAVING  #$(COMMON_O)

clean: 
	rm -f WRITE_TO_BITWEAVING  


WRITE_TO_BITWEAVING: write_to_bitweaving.cpp $(COMMON_CPP) 
	${CXX} ${CPP_FLAG} ${CPP_INCLUDE} write_to_bitweaving.cpp $(COMMON_CPP) -o $@ $(CPP_LAST)
