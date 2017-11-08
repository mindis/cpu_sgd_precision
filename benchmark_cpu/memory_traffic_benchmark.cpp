// Copyright (C) 2017 Zeke Wang - Systems Group, ETH Zurich

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//*************************************************************************


#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <immintrin.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define INTEL_PCM_ENABLE


#ifdef INTEL_PCM_ENABLE

#include "perf_counters.h"
struct Monitor_Event inst_Monitor_Event = {
	{
		{0,0},
		{0,0},
		{0,0},
		{0,0},
	},
	0,
	{
		"core_0",
		"core_1",
		"core_2",
		"core_3",
	},
	{
		{0,0},
		{0,0},
		{0,0},
		{0,0},		
	},
	2,
	{
		"MIC_0",
		"MIC_1",
		"MIC_2",
		"MIC_3",
	},
	0	
};

#endif

	
#ifdef __INTEL_COMPILER
typedef long si64;
#else
typedef long long si64;
#endif





#define CACHE_LINE_SIZE   64
#define SIZE_OF_SEL_ARRAY 16*1024



#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>


typedef struct rand_state_32 {
	uint32_t num[625];
	size_t index;
} rand32_t;

rand32_t *rand32_init(uint32_t seed)
{
	rand32_t *state = (rand32_t *) malloc(sizeof(rand32_t));
	uint32_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 623 ; ++i)
		n[i + 1] = 0x6c078965 * (n[i] ^ (n[i] >> 30));
	state->index = 624;
	return state;
}

uint32_t rand32_next(rand32_t *state)
{
	uint32_t y, *n = state->num;
	if (state->index == 624) {
		size_t i = 0;
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i + 397] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 227);
		n[624] = n[0];
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i - 227] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 624);
		state->index = 0;
	}
	y = n[state->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680;
	y ^= (y << 15) & 0xefc60000;
	y ^= (y >> 18);
	return y;
}



unsigned int scan_with_m64(unsigned int *pool_addr, unsigned short *sel_array, uint64_t num_loop, float sel)
{
	uint32_t literal = (uint32_t) ( sel*(float)(1<<16) );
	uint64_t sum     = 0;
	uint64_t counter = 0;	

	for (uint64_t i = 0; i < num_loop; i ++)
	{
		if (sel_array[i&(SIZE_OF_SEL_ARRAY-1)] < literal) //The sel determines whether to read the data from memory....
		{
			sum += ((uint64_t*)pool_addr)[i];
			counter++;
		}
	}
	printf("real selectivity is %f\n", (double)counter/(double)num_loop );
	return sum;
}


unsigned int scan_with_avx(unsigned int *pool_addr, unsigned short *sel_array, uint64_t num_loop, float sel)
{
	uint32_t literal = (uint32_t) ( sel*(float)(1<<16) );
	__m128i      sum = _mm_set1_epi8(0); 	
	uint64_t counter = 0;	

	for (uint64_t i = 0; i < num_loop; i ++)
	{
		if (sel_array[i&(SIZE_OF_SEL_ARRAY-1)] < literal) //The sel determines whether to read the data from memory....
		{
			sum = _mm_add_epi32 (sum, _mm_loadu_si128( (__m128i *)pool_addr + i ) ); //  ((uint64_t*)pool_addr)[i];
			counter++;
		}
	}

	unsigned int avx_array[16];
	_mm_storeu_si128( (__m128i *)avx_array, sum);

	printf("real selectivity is %f\n", (double)counter/(double)num_loop );

	return avx_array[0];
}

unsigned int scan_with_avx2(unsigned int *pool_addr, unsigned short *sel_array, uint64_t num_loop, float sel)
{
	uint32_t literal = (uint32_t) ( sel*(float)(1<<16) );
	__m256i      sum = _mm256_set1_epi8(0); 	
	uint64_t counter = 0;	

	for (uint64_t i = 0; i < num_loop; i ++)
	{
		if (sel_array[i&(SIZE_OF_SEL_ARRAY-1)] < literal) //The sel determines whether to read the data from memory....
		{
			sum = _mm256_add_epi32 (sum, _mm256_loadu_si256( (__m256i *)pool_addr + i ) ); //  ((uint64_t*)pool_addr)[i];
			counter++;
		}
	}

	unsigned int avx_array[16];
	_mm256_storeu_si256( (__m256i *)avx_array, sum);

	printf("real selectivity is %f\n", (double)counter/(double)num_loop );

	return avx_array[0];
}






void main(int argc, char **argv)
{
	float       sel			= argc > 1 ? atof(argv[1]) : 0.0; //Running all the case with different selectivites.
	int         bits		= argc > 2 ? atoi(argv[2]) : 64;  //deflaut to use 64-bit.
   
	uint64_t tuples			= 1000000000; //The total size of tuples should be large.
  
	printf("The number of bits to be examined is : %d\n", bits);
	printf("The selectivity to be examined is :    %f\n", sel);

    

	/* initialize random seed: */
	srand (time(NULL));
	int seed                   = rand();
    rand32_t *gen              = rand32_init(seed);

	//This part is used to determine the selectivity.
	unsigned short *sel_array = (unsigned short *)aligned_alloc(CACHE_LINE_SIZE, sizeof(short)*SIZE_OF_SEL_ARRAY);
	/* Assign the selection array with random values... */
	for (int i = 0; i < SIZE_OF_SEL_ARRAY; i++)
		sel_array[i] = rand32_next(gen) & 65535; //rand() & 65535; 

	//This part is used to initilization of pool.
	unsigned int *pool_addr =  (unsigned int *)aligned_alloc(CACHE_LINE_SIZE, tuples * sizeof(unsigned int) );
	for (uint64_t i = 0; i < tuples; i++)
		pool_addr[i] = i&65535; 


	//The real number of loop.
	uint64_t num_loop; 
	unsigned int result; 

	#ifdef INTEL_PCM_ENABLE		
        PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
        PCM_start();
    #endif	

	if (bits == 64)
	{
		num_loop = tuples/2;
		result = scan_with_m64(pool_addr, sel_array, num_loop, sel);
	}
    else if (bits == 128)
	{
		num_loop = tuples/4;
		result = scan_with_avx(pool_addr, sel_array, num_loop, sel);
	}	else if (bits == 256)
	{
		num_loop = tuples/8;
		result = scan_with_avx2(pool_addr, sel_array, num_loop, sel);
	}
	else 
		printf("wrong number of bits. The supported number of bits: 64, 128, 256.\n");

	#ifdef INTEL_PCM_ENABLE		
        PCM_stop();
        printf("=====print the profiling result==========\n");//PCM_log("======= Partitioning phase profiling results ======\n");
        PCM_printResults();		
		PCM_cleanup();
    #endif	

	printf("The result is %d. The value is not important.....\n", result);

	free(pool_addr);
	free(sel_array);

}



