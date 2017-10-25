// Copyright 2012 Chris Re, Victor Bittorf
//
 //Licensed under the Apache License, Version 2.0 (the "License");
 //you may not use this file except in compliance with the License.
 //You may obtain a copy of the License at
 //    http://www.apache.org/licenses/LICENSE-2.0
 //Unless required by applicable law or agreed to in writing, software
 //distributed under the License is distributed on an "AS IS" BASIS,
 //WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 //See the License for the specific language governing permissions and
 //limitations under the License.

// The Hazy Project, http://research.cs.wisc.edu/hazy/
// Author : Victor Bittorf (bittorf [at] cs.wisc.edu)

#ifndef HAZY_VECTOR_VECTOROPS_INL_H
#define HAZY_VECTOR_VECTOROPS_INL_H
#include "string.h"

#include "hazy/util/sort.h"
#include "hazy/vector/operations.h"

// See hazy/vector/operations.h for documentation
#include <immintrin.h>


namespace hazy {
namespace vector {

void inline Convert_from_bitweaving(FVector<unsigned char> & dest, FVector<unsigned int> src, unsigned num_bits) 
{
	uint64_t numFeatures    = dest.size;
	unsigned char* vec_char = dest.values;
	unsigned int* vec_int   = src.values;
	
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

	uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;
	
	for (size_t i = 0; i < numFeatures; i++) 
  	{
    //vec_char[i] = extract_from_bitweaving(src.values, i, numFeatures);
	
	  //Compute the main part of numFeatures.
	  if (i < num_features_main)
	  {
		uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE	  ) * BITS_OF_ONE_CACHE_LINE; //
		uint32_t int_offset  = ( i&(BITS_OF_ONE_CACHE_LINE-1) )/32;
		uint32_t bit_offset  = i & 31;
	
		//The next 32 CLs contains the information of the feature. 
		unsigned char result = 0;
		unsigned int tmp;
		for (uint32_t j = 0; j < num_bits; j++)
		{
								//main		  bit	 which ints 
		  tmp	  = vec_int[main_offset + 16 * j + int_offset]; 
		  result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		}
		vec_char[i] = result; //return result;
	  }
	  else
	  {
		uint32_t num_r_f = numFeatures - num_features_main;
	
		if (num_r_f <= 64)												 //////remainder <= 64
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i & (64-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 2 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 128)										  //////64 < remainder <= 128
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(128-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 4 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f <= 256)										  //////128 < remainder <= 256
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(256-1) )/32;
		  uint32_t bit_offset  = i & 31;
	
		  //The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 8 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}
		else if (num_r_f < 512) 										 //////256 < remainder < 512
		{ 
		  uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
		  uint32_t int_offset  = ( i&(512-1) )/32;
		  uint32_t bit_offset  = i & 31;
		//The next 32 CLs contains the information of the feature. 
		  uint32_t result = 0;
		  uint32_t tmp;
		  for (uint32_t j = 0; j < num_bits; j++)
		  {
							  //main		  bit	 which ints 
			tmp 	= vec_int[main_offset + 16 * j + int_offset]; 
			result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (7-j)); //
		  }
		  vec_char[i] = result; //return result;
		}			
	  }
    
  }
}

//add src to the destination. 
template <typename T>
void inline avg_list(FVector<T> & dest, FVector<T> *src, unsigned N) {
  const uint64_t n0         = dest.size;
  const uint64_t n1         = (n0 >> 4) << 4;

  T scale_factor            = 1.0 /(T)N;
  const __m256 scale_const  = _mm256_set1_ps(1.0 /(T)N); //T scale_factor = 1.0 /(T)N;

  for (size_t i = 0; i < n1; i+= 32) 
  {
    //T sum     = 0.0;
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();    
    //for (unsigned j = 0; j < N; j++)  
    //  sum += (src[j])[i];
    for (unsigned j = 0; j < N; j++)
    {
      __m256 v1 = _mm256_loadu_ps( (float const *)&((src[j])[i + 0 ]) );
      __m256 v2 = _mm256_loadu_ps( (float const *)&((src[j])[i + 8 ]) );
      __m256 v3 = _mm256_loadu_ps( (float const *)&((src[j])[i + 16]) );
      __m256 v4 = _mm256_loadu_ps( (float const *)&((src[j])[i + 24]) );      
      sum1      = _mm256_add_ps ( sum1, v1 );
      sum2      = _mm256_add_ps ( sum2, v2 );
      sum3      = _mm256_add_ps ( sum3, v3 );
      sum4      = _mm256_add_ps ( sum4, v4 );
    }  

    //dest[i] = sum * scale_factor;
    _mm256_storeu_ps((float *)(&dest[i + 0 ]), _mm256_mul_ps(sum1, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 8 ]), _mm256_mul_ps(sum2, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 16]), _mm256_mul_ps(sum3, scale_const) );
    _mm256_storeu_ps((float *)(&dest[i + 24]), _mm256_mul_ps(sum4, scale_const) );
  }

  for (size_t i = n1; i < n0; i++)
  {
    T sum   = 0.0;
    for (unsigned j = 0; j < N; j++)  
      sum  += (src[j])[i];

    dest[i] = sum * scale_factor;
  }

}


//add src to the destination. streaming load from src, no tag for the source...
template <typename T>
void inline add(FVector<T> & dest, FVector<T> const &src) {
  unsigned vix = 0;
  unsigned i = 0;
  for (size_t i = 0; i < dest.size; i++) {
    dest.values[i]  += dest.values[i] + src.values[i];
  }
}

//add src to the destination. 
template <typename T>
void inline add_mult(FVector<T> & dest, FVector<T> const &src1, FVector<T> const &src2, T scale_factor) {
  unsigned vix = 0;
  unsigned i = 0;
  for (size_t i = 0; i < dest.size; i++) {
    dest.values[i]  += (src1.values[i] + src2.values[i])*scale_factor;
  }
}


template <typename T>
bool IsValid(SVector<T> const &v) {
  for (size_t i = 0; i < v.size; i++) {
    if (v.index[i] < 0) { assert(false); return false; }
    if (v.index[i] >= v.size) { assert(false); return false; }
  }
  return true;
}


template <typename T>
double inline Norm2(FVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return norm;
}

template <typename T>
double inline Norm2WithoutSquare(FVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return sqrt(norm);
}

template <typename T>
double inline Norm2WithoutSquare(SVector<T> const &v) {
  double norm = 0;
  for (size_t i = 0; i < v.size; i++) {
    norm += v.values[i] * v.values[i];
  }
  return sqrt(norm);
}


template <typename T, typename int_T>
void inline Project(SVector<T> const& v, FVector<int_T> const &indexes, T *out) {
  unsigned vix = 0;
  unsigned i = 0;
  while (static_cast<unsigned>(i) < indexes.size) {
    if (vix >= v.size) {
      out[i++] = 0;
      continue;
    }
    if (indexes.values[i] > v.index[vix]) {
      vix++;
      continue;
    }
    if (indexes.values[i] < v.index[vix]) {
      out[i++] = 0;
      continue;
    }
    assert(indexes.values[i] == v.index[vix]);
    out[i++] = v.values[vix];
  }
}

template <typename float_t>
void inline SimplexProject(FVector<float_t> &vec) {
  float_t v[vec.size];
  float_t *vecf = vec.values;
  for (unsigned i = 0; i < vec.size; i++) {
    v[i] = vecf[i];
  }
  util::QuickSort(v, vec.size);
  //std::vector<float_t> v(vec.values, &vec.values[vec.size]);
  //std::sort(v.begin(), v.end());
  int i     = vec.size-2;
  double ti = 0.0, ti_sum = 0.0;

  while (i >= 0) {
    ti_sum += v[i+1];
    ti  = (ti_sum - 1)/(vec.size - 1 -i);
    if(ti > v[i]) break;
    i--;
  }

  for (unsigned k = 0; k < vec.size; k++) {
    vec.values[k] = std::max(0.0, vec.values[k] - ti);
  }
}

// Only apply threshold to the masked entries
void MaskThresholdZero(vector::FVector<double> &x,
                       const vector::FVector<size_t> &mask) {
  for (size_t i = 0; i < mask.size; i++) {
    if (x.values[mask.values[i]] <= 0) {
      x.values[mask.values[i]]  = 0;
    }
  }
}

//template <typename float_t>
void inline Zero(FVector<float> &v) {
  float_t *  outv           = v.values;

  const uint64_t n0         = v.size;
  const uint64_t n1         = (n0 >> 4) << 4;
  
  //printf("In the function Zero, with SIMD\n");

  const __m256 zero_v       = _mm256_set1_ps(0.0); //T scale_factor = 1.0 /(T)N;
  float_t scale_factor            = 0.0;

  for (size_t i = 0; i < n1; i+= 32) 
  {
    //dest[i] = sum * scale_factor;
    _mm256_storeu_ps((float *)(outv + i + 0 ), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 8 ), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 16), zero_v );
    _mm256_storeu_ps((float *)(outv + i + 24), zero_v );
  }

  for (size_t i = n1; i < n0; i++)
  {
    outv[i] = scale_factor;
  }  

}


template <typename float_t>
void inline Zero(FVector<float_t> &v) {
  float_t *  outv           = v.values;
  const uint64_t n0         = v.size;
  //printf("In the function Zero, with Scalar\n");

  for (size_t i = 0; i < n0; i++)
  {
    outv[i] = (float_t)0;
  }  

}

template <typename float_t>
void inline Zero(SVector<float_t> &v) {
  for (unsigned i = 0; i < v.size; i++) {
    v.values[i] = 0;
  }
}

template <typename float_t>
void inline ThresholdZero(SVector<float_t> &v) {
  for (unsigned i = 0; i < v.size; i++) {
    if (v.values[i] < 0) 
      v.values[i] = 0;
  }
}

template <typename float_u>
void inline CopyInto(FVector<float_u> const &u, FVector<float_u> &out) {
  float_u *  outv           = out.values;
  float_u *  inv            = u.values;

  const uint64_t n0         = out.size;
  const uint64_t n1         = (n0 >> 4) << 4;


  for (size_t i = 0; i < n1; i+= 32) 
  {
    __m256 v1 = _mm256_loadu_ps( (float const *) (inv + i + 0)  );
    __m256 v2 = _mm256_loadu_ps( (float const *) (inv + i + 8)  );
    __m256 v3 = _mm256_loadu_ps( (float const *) (inv + i + 16) );
    __m256 v4 = _mm256_loadu_ps( (float const *) (inv + i + 24) );      

    _mm256_storeu_ps((float *)(outv + i + 0 ), v1 );
    _mm256_storeu_ps((float *)(outv + i + 8 ), v2 );
    _mm256_storeu_ps((float *)(outv + i + 16), v3 );
    _mm256_storeu_ps((float *)(outv + i + 24), v4 );
  }

  for (size_t i = n1; i < n0; i++)
  {
    outv[i] = inv[i];
  }

}


} // namespace vector
} // namespace hazy
#endif
