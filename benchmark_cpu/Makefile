

CC=icc
OPT=  -O3 -pthread -fPIC -std=c++11 -fopenmp -lrt 
COMMONTOOL = libpcm_2_11.a


all: TEST


TEST: memory_traffic_benchmark.cpp
	$(CC) $(OPT) $< -o $@ #$(COMMONTOOL) 
	
	
clean:
	rm -rf TEST