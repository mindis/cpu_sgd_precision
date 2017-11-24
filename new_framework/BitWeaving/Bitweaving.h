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

#ifndef BITWEAVING_H
#define BITWEAVING_H

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
#include <vector>


#include "sample.h"
#include "hazy/types/entry.h"
#include "hazy/vector/fvector.h"

#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif


//#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512
typedef float fp_type;     


////////////////////////////////////////////////////////////////////
class BitWeavingBase
{
public:
	//////////////Constructor///////////////
			 BitWeavingBase(){};
	explicit BitWeavingBase(const char *fname, uint32_t dimension, uint32_t num_bits, uint32_t num_samples, bool huge_table_en);
	explicit BitWeavingBase(uint32_t dimension, uint32_t num_bits, uint32_t num_samples, bool huge_table_en);
	//////////////Destructor///////////////
			~BitWeavingBase();

	void     statistic_show();

	// It is called by the pre-processing stage, where we compress the training dataset into BitWeaving.
	// Write one sample to the BitWeaving memory, all the 32-bits... (dimension, index)
	// dimension: Need to check whether the number of features is legal...
	// samp_index: the index of sample to stored into the 
	void	write_to_bitweaving(int samp_index, hazy::vector::FVector<unsigned int> samp, float rating);	

	//Read one sample from the BitWeaving memory, only a few bits...
	void	read_from_bitweaving(int samp_index, LinearModelSampleBitweaving &samp);

	template <class Scan>  
	size_t  write_file_to_bitweaving(Scan &scan, uint32_t dimension, uint32_t num_samples);

private:
	bool          fd_en;
	int 		  zk_fd;		 //File id for the input dataset...
	uint64_t	  zk_total_len;  //Size of this memory (bytes)...
	uint32_t     *zk_mem_addr;   //Mmap the fd (MAP_SHARED) to the memory address or MAP_ANONYMOUS...	
	float        *zk_rating_addr;//Store the ratings...

	///Organization of mmap area///
	uint32_t      zk_num_regions;//Number of memory regions ...	
	uint64_t	  zk_region_offset;

	uint32_t      zk_num_bits;   //Number of bits in each memory region. 		
	uint32_t      zk_align_bits; //Number of bits for each bit of the remainder. 		

	////Information of input training dataset....////
	uint32_t      zk_dimension;  //Dimension of each sample...		
	uint32_t      zk_num_samples;//Number of samples...		
	uint32_t      zk_CLs_a_sample; 
	uint32_t compute_CLs_per_sample(void);
};

template <class Scan>  
size_t BitWeavingBase::write_file_to_bitweaving(Scan &scan, uint32_t dimension, uint32_t num_samples)
{
	//std::vector<LinearModelSample> examps;
	int lastrow = -1;
	double rating = 0.0;
	std::vector<uint32_t> data;
	std::vector<int> index;

	int max_col = 0;
	int samp_index = 0;

    // set to zero

	while (scan.HasNext()) 
	{
		const hazy::types::Entry &e = scan.Next();
		if (lastrow == -1) {
        	lastrow = e.row;
		}
		if ((lastrow != e.row) || (!scan.HasNext())) 
		{
        // finish off the previous vector and start a new one
			lastrow = e.row;

			if(!scan.HasNext()) 
			{
				data.push_back( (uint32_t)(e.rating * 4294967295.0) );
				index.push_back(e.col);
			}

			uint32_t *zeros  = new uint32_t[dimension];
			hazy::vector::FVector<uint32_t> temp_vector(zeros, dimension);

			//Assign the value to the destination fvector which is written to ....
			hazy::vector::Zero(temp_vector);
			for (size_t j = 0; j < data.size(); j++) 
			{
				zeros[ index[j] ] = data[j];
        	}			

        	//Write the sample to the disk...
			if (samp_index >= num_samples)
				printf("samp_index = %d, num_samples = %d", samp_index, num_samples);
			
			assert(samp_index < num_samples);

			write_to_bitweaving(samp_index, temp_vector, (float)rating);
/*
			LinearModelSampleBitweaving bitweaving_samp;
			read_from_bitweaving(samp_index, bitweaving_samp);
			unsigned char *dest = (unsigned char *)malloc(2*dimension);
			hazy::vector::FVector<unsigned char> dest_char_vector (dest, dimension);;
			bitweaving_samp.Unpack_from_bitweaving(dest_char_vector, 8);
			for (int ii = 0; ii < dimension; ii++)
			{
				if (dest_char_vector[ii] != (zeros[ii] >>24) )
				{	
					printf("samp_index = %d\n", samp_index);
					printf("The data is not right!!!!!! dest[%d] = 0x%x, original = 0x%x\n", ii, dest_char_vector[ii], zeros[ii]);
					return 2;
				}
			}
			free (dest);
*/
			samp_index++;

			delete zeros;			
			//LinearModelSample temp(rating, d, i, data.size(), dimension);
			//examps.push_back(temp);
			rating = 0.0;
			data.clear();
			index.clear();
		}

		if (e.col < 0) 
		{
			rating = e.rating;
		} else 
		{
			if (e.col > max_col) 
			{
				max_col = e.col;
			}
			data.push_back( (uint32_t)(e.rating * 4294967295.0) );
			index.push_back(e.col);
		}
    }
/*
    // Copy from temp vector into persistent memory
    ex.size = examps.size();
    ex.values = new LinearModelSample[ex.size];
    for (size_t i = 0; i < ex.size; i++) {
      new (&ex.values[i]) LinearModelSample(examps[i]);
    }
*/    
    return max_col+1;
  }

//BitWeavingBase::BitWeavingBase(){ }

BitWeavingBase::BitWeavingBase(const char *fname, uint32_t dimension, uint32_t num_bits, uint32_t num_samples, bool huge_table_en)
{
	zk_dimension   = dimension;
	zk_num_bits    = num_bits;
	zk_num_samples = num_samples;

	assert ( (num_bits == 2)||(num_bits == 4)||(num_bits == 8) ); //assert the validation of bits.
	zk_num_regions            =  32/num_bits;
	uint32_t least_align_bits = 512/num_bits; 

	uint32_t remainder_features   = zk_dimension & (BITS_OF_ONE_CACHE_LINE - 1); 

	if (remainder_features == 0)
		zk_align_bits =  0;
	else if ( (remainder_features <=  64) && (least_align_bits <=  64) )
		zk_align_bits =  64; //rem_num =  4;
	else if ( (remainder_features <= 128) && (least_align_bits <= 128) )
		zk_align_bits =  128; //rem_num =  8;
	else if ( (remainder_features <= 256) && (least_align_bits <= 256) )
		zk_align_bits =  256; //rem_num =  16;
	else	
		zk_align_bits =  512; //rem_num =  32;


	zk_CLs_a_sample    = BitWeavingBase::compute_CLs_per_sample();//(dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32 + 
	zk_total_len       = zk_CLs_a_sample * 16 * num_samples;
	zk_region_offset   = zk_total_len / zk_num_regions;
	//Open the file which contains the training dataset...
	zk_fd = open(fname, O_RDWR);
	if (zk_fd == NULL) {
		printf("Could not open file: %s\n", fname);
		//return;
	}

	fd_en = true;
	if (huge_table_en)
    	zk_mem_addr =  (uint32_t *)mmap (0, (zk_total_len+num_samples) * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_HUGETLB, zk_fd, 0); //MAP_SHARED
	else 
    	zk_mem_addr =  (uint32_t *)mmap (0, (zk_total_len+num_samples) * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_SHARED, zk_fd, 0); //|MAP_HUGETLB
	//Try to mapp the file to the memory region (zk_mem_addr, zk_total_len).
    if (zk_mem_addr == MAP_FAILED) 
    {
		perror ("mmap error");
    }
    else
    	printf("Succesfully map the file to the memory location!!!\n");


    zk_rating_addr = (float *)(zk_mem_addr + zk_total_len); 
}

BitWeavingBase::BitWeavingBase(uint32_t dimension, uint32_t num_bits, uint32_t num_samples, bool huge_table_en)
{
	zk_dimension      = dimension;
	zk_num_bits       = num_bits;
	zk_num_samples    = num_samples;

	assert ( (num_bits == 2)||(num_bits == 4)||(num_bits == 8) ); //assert the validation of bits.

	zk_align_bits      = 512/num_bits;
	zk_num_regions     =  32/num_bits;

	zk_CLs_a_sample    = BitWeavingBase::compute_CLs_per_sample();//(dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32 + 
	zk_total_len       = zk_CLs_a_sample * 16 * num_samples;
	zk_region_offset   = zk_total_len/zk_num_regions;
	//Open the file which contains the training dataset...

	fd_en = false;
	if (huge_table_en)
    	zk_mem_addr =  (uint32_t *)mmap (0, (zk_total_len+num_samples) * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0); //  
	else 
    	zk_mem_addr =  (uint32_t *)mmap (0, (zk_total_len+num_samples) * sizeof(uint32_t), PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0); // | MAP_HUGETLB 
	//Try to mapp the file to the memory region (zk_mem_addr, zk_total_len).
    if (zk_mem_addr == MAP_FAILED) {
		perror ("mmap error");
		//return 1;
    }
    else
    	printf("Succesfully map the file to the memory location!!!\n");
}

BitWeavingBase::~BitWeavingBase()
{
    if ( munmap(zk_mem_addr, (zk_total_len+zk_num_samples) * sizeof(uint32_t)) != 0)
        printf("There is error when doing munmap (mem_addr, total_len)\n");
    else 
        printf("munmap is successful\n");
}

void BitWeavingBase::read_from_bitweaving(int samp_index, LinearModelSampleBitweaving &samp)
{
	assert( (samp_index >= 0)&& (samp_index < zk_num_samples) );

	samp.rating        = zk_rating_addr[samp_index]; //fix it..

	samp.sample_addr   = (uint32_t *) zk_mem_addr + samp_index*(zk_CLs_a_sample*16)/zk_num_regions;
	samp.num_bits      = zk_num_bits  ;
	samp.region_offset = zk_region_offset;
	samp.align_bits    = zk_align_bits   ;
	samp.dimension     = zk_dimension    ;
}

void BitWeavingBase::write_to_bitweaving(int samp_index, hazy::vector::FVector<uint32_t> samp, float rating)
{
	assert(zk_dimension == samp.size);  //Check whether the dimension matchs or not;
	assert( (samp_index >= 0) && (samp_index < zk_num_samples) );//Check whether the index of sample is valid or not;

	uint32_t numFeatures         = zk_dimension;
	uint32_t num_features_main 	 = (numFeatures/BITS_OF_ONE_CACHE_LINE)*BITS_OF_ONE_CACHE_LINE; 
	uint32_t *src                = (uint32_t *)samp.values;

	zk_rating_addr[samp_index]   = rating;

	//zk_region_offset
	uint64_t sample_offset       = (samp_index*zk_CLs_a_sample*16)/zk_num_regions; //Sample offset....

	//Deal with the main part of dr_numFeatures.
	for (uint32_t j = 0; j < num_features_main; j += BITS_OF_ONE_CACHE_LINE)
	{
		uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};

		uint64_t feature_offset = j/zk_num_regions; 		//Feature offset...

		//1: initilization off tmp buffer..
		for (uint32_t k = 0; k < BITS_OF_ONE_CACHE_LINE; k += 8)
		{
			__m256i v1 = _mm256_loadu_si256( (__m256i const *)&(src[j+k]) );
			_mm256_storeu_si256( (__m256i *)&(tmp_buffer[k]), v1);
		}  

		//2: focus on the k-th bit...
		for (int k = 0; k < 32; k++)
		{ 
			unsigned char result_buffer[BITS_OF_ONE_CACHE_LINE/8] = {0};	//16 ints == 512 bits...
			//2.1: re-order the data according to the bit-level...
			for (int m = 0; m < BITS_OF_ONE_CACHE_LINE; m+=8)
			{
				__m256i v_data	   = _mm256_loadu_si256((__m256i const *)&tmp_buffer[m]);
				int tmp 		   = _mm256_movemask_ps( _mm256_castsi256_ps(v_data) );
				result_buffer[m/8] = (unsigned char)tmp;
				v_data			   = _mm256_slli_epi32(v_data, 1);
				_mm256_storeu_si256((__m256i *)&tmp_buffer[m], v_data);
			}

			uint64_t region_offset = zk_region_offset * (k/zk_num_bits);
			uint64_t bit_offset    =               16 * (k%zk_num_bits);

			//2.2: store the bit-level result back to the memory...
			__m256i v_data_1       = _mm256_loadu_si256((__m256i const *)&result_buffer[0]);
			__m256i v_data_2       = _mm256_loadu_si256((__m256i const *)&result_buffer[32]);

			//Depending on the samp_index, calculate the offset....
														//Sample        512 features  region        bit
			_mm256_storeu_si256((__m256i *)&zk_mem_addr[sample_offset+feature_offset+region_offset+bit_offset+0], v_data_1);
			_mm256_storeu_si256((__m256i *)&zk_mem_addr[sample_offset+feature_offset+region_offset+bit_offset+8], v_data_2);
		}
	}

	//handle the remainder....
	//Deal with the remainder of features, with the index from j...
	uint32_t num_r_f        = numFeatures - num_features_main;
	uint64_t feature_offset = num_features_main/zk_num_regions; 

	if (num_r_f > 0)
	{
		uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
		for (int k = 0; k < num_r_f; k++)
			tmp_buffer[k] = src[num_features_main + k]; //j is the existing index...
		
		for (int k = 0; k < 32; k++) //64 bits for each bit...
		{
			uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
			for (int m = 0; m < num_r_f; m++)
			{
				result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
				tmp_buffer[m] 	    = tmp_buffer[m] << 1; 	  
			}

			uint64_t region_offset  = (zk_region_offset) * (k/zk_num_bits);
			uint64_t bit_offset     = (zk_align_bits/32) * (k%zk_num_bits);

			//Each bit contains "zk_align_bits" bits. 
			for (int m = 0; m < (zk_align_bits/32); m++)
			{
				zk_mem_addr[sample_offset+feature_offset+region_offset+bit_offset+m] = result_buffer[m];
			}
		} 			
	}
}

void BitWeavingBase::statistic_show() 
{
    printf("===============================================================\n");
    printf("========================BitWeavingBase=========================\n");
    printf("===============================================================\n");

    printf("dimension   = %d\n", zk_dimension);
    printf("num_samples = %d\n", zk_num_samples);
    printf("num_bits    = %d\n", zk_num_bits);
    printf("align_bits  = %d\n", zk_align_bits);
    printf("num_regions = %d\n", zk_num_regions);

    printf("region_offset = 0x%lx\n", zk_region_offset);
    printf("total_len     = 0x%lx\n", zk_total_len); 
    printf("CLs_a_sample  = %ld\n", zk_CLs_a_sample); 
}

uint32_t BitWeavingBase::compute_CLs_per_sample(void) 
{
	  //With the chunk of 512 features...
	uint32_t main_num 		      = (zk_dimension/BITS_OF_ONE_CACHE_LINE)*32; //It is for CLs
	uint32_t rem_num			  = zk_align_bits*32/512;
	//For the remainder of zk_dimension...

/*	
	uint32_t remainder_features   = zk_dimension & (BITS_OF_ONE_CACHE_LINE - 1); 

	if (remainder_features == 0)
		rem_num =  0;
	else if ( (remainder_features <=  64) && (zk_align_bits <=  64) )
		rem_num =  4;
	else if ( (remainder_features <= 128) && (zk_align_bits <= 128) )
		rem_num =  8;
	else if ( (remainder_features <= 256) && (zk_align_bits <= 256) )
		rem_num = 16;
	else	
		rem_num = 32;
*/
	return main_num + rem_num;
}

/*
class training_dataset_manager_write: public training_dataset_manager_base
{
public:
	explicit training_dataset_manager_write(const char *fname, uint64_t dimension, uint64_t num_samples);

	explicit training_dataset_manager_write(uint64_t dimension, uint64_t num_samples );
}
*/




#endif
