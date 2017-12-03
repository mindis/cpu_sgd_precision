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

#ifndef BITWEAVING_SAMPLE_H
#define BITWEAVING_SAMPLE_H

#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef __linux__
    #include <malloc.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>
#include <assert.h>

#include "hazy/vector/fvector.h"


//#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512


//////////////////////////////////////////float//////////////////////////////////////////////////////////
//With the information from this struct, we can unpack the samp with different bits.
struct LinearModelSampleBitweaving
{
  float         rating;           //rating of this sample...
  uint32_t     *sample_addr;      //The base address of this sample.
  uint32_t      num_bits;         //Number of memory regions
  uint64_t      region_offset;    //The address offset bwtween two conjunctive memory regions.
  uint32_t      align_bits;       //The aligned bits for the remainder...
  uint32_t      dimension;        //Dimension of each sample.

  void print_info()
  {
    printf("===============================================================\n");

  	printf("rating      = %f\n", rating);
  	printf("dimension   = %d\n", dimension);
  	printf("align_bits  = %d\n", align_bits);
  	printf("num_bits    = %d\n", num_bits);
  	printf("region_offset = 0x%lx\n", region_offset);
  }

void inline Unpack_from_bitweaving(hazy::vector::FVector<unsigned short> & dest, unsigned bits) 
{
	size_t   numFeatures      = dimension;
	uint32_t       * src      = sample_addr;		
	unsigned short  *vec_short= (unsigned short  *)dest.values;

	__m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
	__m256i v_mask	 = _mm256_set1_epi32(0x01010101);
	__m256i v_sum, v_data, v_data_1;
	__m256i v_high, v_low;
			

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
			
	for (size_t base = 0; base < numFeatures; base += BITS_OF_ONE_CACHE_LINE) 
	{	
		size_t base_r          = base/(32/num_bits);
		uint32_t num_r_f       = numFeatures - num_features_main;
		uint32_t num_iteration; // << 5 
		uint32_t stride        = 0;

		if (base < num_features_main)
		{
			stride             = 16;
			num_iteration      = 16;
		}
		else
		{   
			num_iteration = ((num_r_f+31)>>5); // << 5 
			stride        = align_bits/32;		}

		for (size_t offset = 0; offset < num_iteration; offset++)
		{
			//if (offset < bits)
			//{
			//	_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_NTA);	//Stay at L1
				//_mm_prefetch((char *)(&src[base + 2*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T2);//Stay at L1
			//}		
			v_sum = _mm256_set1_epi32(0);
			unsigned int data_src;
				//printf("In the inner loop\n");
	
			data_src = src[base_r + (0/num_bits)*region_offset + (0%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

			data_src = src[base_r + (1/num_bits)*region_offset + (1%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );

			data_src = src[base_r + (2/num_bits)*region_offset + (2%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );

			data_src = src[base_r + (3/num_bits)*region_offset + (3%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );

			data_src = src[base_r + (4/num_bits)*region_offset + (4%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );

			data_src = src[base_r + (5/num_bits)*region_offset + (5%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );

			data_src = src[base_r + (6/num_bits)*region_offset + (6%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );

			data_src = src[base_r + (7/num_bits)*region_offset + (7%num_bits)*stride + offset];
			v_data	 =	_mm256_set1_epi32(data_src); 
			v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
			v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
			v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );

					//unsigned char sum_array_high[64]; //32 is enough.
					//_mm256_store_si256((__m256i *)sum_array_high, v_sum);
			v_high = v_sum;
			v_sum = _mm256_set1_epi32(0);

					data_src = src[base_r + (8/num_bits)*region_offset + (8%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );

			if (bits >=10)
			{
					data_src = src[base_r + (9/num_bits)*region_offset + (9%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
			 if (bits >=11)
			 {
					data_src = src[base_r + (10/num_bits)*region_offset + (10%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );
			  if (bits >=12)
			  {
					data_src = src[base_r + (11/num_bits)*region_offset + (11%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );
				if (bits >=13)
				{
					data_src = src[base_r + (12/num_bits)*region_offset + (12%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );
				if (bits >=14)
				{
					data_src = src[base_r + (13/num_bits)*region_offset + (13%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );
				 if (bits >=15)
				 {
					data_src = src[base_r + (14/num_bits)*region_offset + (14%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );
				  if (bits >=16)
				  {
					data_src = src[base_r + (15/num_bits)*region_offset + (15%num_bits)*stride + offset];
					v_data	 =	_mm256_set1_epi32(data_src); 
					v_data	 =	_mm256_srav_epi32(v_data, v_offset); //shift it...
					v_data_1 =	_mm256_and_si256 (v_data, v_mask  ); //3  v_data
					v_sum	 =	_mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );
				  }}}}}} }	

				v_low = v_sum;

			__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0,
			 				                              15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0);
    		__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                		5,  1, 4,  0); 

			__m256i v_data_2        = _mm256_shuffle_epi8(v_low, v_shuffle_constant);
			__m256i v_low_tmp       = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);


			       v_data_2         = _mm256_shuffle_epi8(v_high, v_shuffle_constant);
			__m256i v_high_tmp      = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);



				//Now, we have v_high (3, 1), v_low (2, 0)... objective: (3, 2), (1, 0)...
				__m256i v_data_low  = _mm256_unpacklo_epi8(v_low_tmp, v_high_tmp);				
				__m256i v_data_high = _mm256_unpackhi_epi8(v_low_tmp, v_high_tmp);

				_mm256_storeu_si256((__m256i *)(&vec_short[base + offset*32 +  0]), v_data_low );				
				_mm256_storeu_si256((__m256i *)(&vec_short[base + offset*32 + 16]), v_data_high);				


				__m128i v_data_128_low  = _mm_loadu_si128((__m128i*)(&vec_short[base + offset*32 + 16]) );
				__m128i v_data_128_high = _mm_loadu_si128((__m128i*)(&vec_short[base + offset*32 + 8]) );

				_mm_storeu_si128((__m128i *)(&vec_short[base + offset*32 +  8]), v_data_128_low );				
				_mm_storeu_si128((__m128i *)(&vec_short[base + offset*32 + 16]), v_data_128_high);	
		}
	}
}

void Unpack_from_bitweaving(hazy::vector::FVector<unsigned char> & dest, unsigned bits) 
{
	size_t   numFeatures    = dimension;
	uint32_t       * src    = sample_addr;		
	unsigned char  *vec_char= (unsigned char  *)dest.values;

	__m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
	__m256i v_mask   = _mm256_set1_epi32(0x01010101);
	__m256i v_sum, v_data, v_data_1, v_data_2, v_data_3, v_data_4, v_data_5, v_data_6, v_data_7, v_data_0;

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;

	//For each 512-code chunk
	for (size_t base = 0; base < numFeatures; base += BITS_OF_ONE_CACHE_LINE) 
	{
		size_t base_r          = base/(32/num_bits);
		uint32_t num_r_f       = numFeatures - num_features_main;
		uint32_t num_iteration; // << 5 
		uint32_t stride        = 0;
		if (base < num_features_main)
		{
			stride             = 16;
			num_iteration      = 16;
		}
		else
		{   
			num_iteration = ((num_r_f+31)>>5); // << 5 
			stride        = align_bits/32;
		}

		//size_t base = num_features_main;
		for (size_t offset = 0; offset < num_iteration; offset++)
		{
			//if (offset < num_bits)
			//{
			//	_mm_prefetch((char *)(&src[base + 1*BITS_OF_ONE_CACHE_LINE+ offset*16]), _MM_HINT_T0); //Stay at L1

			// Add the preftch instrcutions for the next 512-element chunk. offset: index of the memory region, . 
			// 
				v_sum = _mm256_set1_epi32(0);
				uint32_t data_src;

				data_src = src[base_r + (0/num_bits)*region_offset + (0%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );
			if (bits >=2)
			{
				data_src = src[base_r + (1/num_bits)*region_offset + (1%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );
			 if (bits >=3)
			 {
				data_src = src[base_r + (2/num_bits)*region_offset + (2%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );
			  if (bits >=4)
			  {
				data_src = src[base_r + (3/num_bits)*region_offset + (3%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );
			   if (bits >=5)
			   {
				data_src = src[base_r + (4/num_bits)*region_offset + (4%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );
			   if (bits >=6)
			   {
				data_src = src[base_r + (5/num_bits)*region_offset + (5%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );
			    if (bits >=7)
			    {
				data_src = src[base_r + (6/num_bits)*region_offset + (6%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );
			    if (bits >=8)
			    {
				data_src = src[base_r + (7/num_bits)*region_offset + (7%num_bits)*stride + offset];
				v_data   =  _mm256_set1_epi32(data_src); 
				v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
				v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
				v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 0) );
			 
			 }}}}}} }

			__m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0,
			 				                              15, 11,  7,  3, 
			 				                              14, 10,  6,  2, 
			 				                              13,  9,  5,  1, 
			 				                              12,  8,  4,  0);
			__m256i v_data_2        = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);
    		__m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                		5,  1, 4,  0); 
			__m256i v_result = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);
			_mm256_store_si256((__m256i *)(&vec_char[base + offset*32]), v_result);
		}
	}
}


};




#endif
