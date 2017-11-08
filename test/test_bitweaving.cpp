#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>       /* time */
#ifdef __linux__
    #include <malloc.h>
#endif

//The related parameters are here. 
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512





//Suppose the size of each value of training dataset is 32-bit, always true for our case...
uint32_t compute_num_CLs_per_sample(uint32_t dr_numFeatures) {
  //With the chunk of 512 features...
  uint32_t main_num           = (dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32; //It is for CLs
  uint32_t rem_num            = 0;

  //For the remainder of dr_numFeatures...
  uint32_t remainder_features = dr_numFeatures & (BITS_OF_ONE_CACHE_LINE - 1); 
  if (remainder_features == 0)
    rem_num = 0;
  else if (remainder_features <= 64)
    rem_num = 4;
  else if (remainder_features <= 128) 
    rem_num = 8;
  else if (remainder_features <= 256) 
    rem_num = 16;
  else  
    rem_num = 32;
  //printf("main_num = %d, rem_num = %d\t", main_num, rem_num);
  return main_num + rem_num;
}

void test_num_CLs()
{
  uint32_t num_features[8] = {1024, 4521, 6512, 541, 214, 5000, 780, 47236};
  for (int i = 0; i < 8; i++)
    printf("Number of features: %d, number of CLs: %d\n", num_features[i], compute_num_CLs_per_sample(num_features[i]));
}







//Now let us test the bitweaving on each sample....
//Make sure the address of "dest" is aligned on the 64-byte boundary. 
void bitweaving_on_each_sample(uint32_t *dest, uint32_t *src, uint32_t numFeatures) 
{
  //Compute the number of CLs for each sample...
  int num_CLs_per_sample     = compute_num_CLs_per_sample(numFeatures);
  //printf("num_CLs_per_sample = %d\n", num_CLs_per_sample);
  //uint32_t *a_fpga_tmp       = a_bitweaving_fpga;
  uint32_t address_index     = 0;
  int num_features_main      = (numFeatures/BITS_OF_ONE_CACHE_LINE)*BITS_OF_ONE_CACHE_LINE;  

  //1, Deal with the main part of dr_numFeatures.
  for (int j = 0; j < num_features_main; j += BITS_OF_ONE_CACHE_LINE)
  {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      //1.1: initilization off tmp buffer..
      for (int k = 0; k < BITS_OF_ONE_CACHE_LINE; k++)
      {
        tmp_buffer[k] = src[j + k];
        //printf("src[%d] = 0x%8x\t", j+k, src[j + k]);
      }  

      //1.2: focus on the data from index: j...
      for (int k = 0; k < 32; k++)
      { 
        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE/32] = {0};  //16 ints == 512 bits...
        //1.2.1: re-order the data according to the bit-level...
        for (int m = 0; m < BITS_OF_ONE_CACHE_LINE; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }
        //1.2.2: store the bit-level result back to the memory...
        dest[address_index++] = result_buffer[0]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[1]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[2]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[3]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[4]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[5]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[6]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[7]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[8]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[9]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[10];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[11];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[12];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[13];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[14];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[15];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
      }
  }

    //Deal with the remainder of features, with the index from j...
    uint32_t num_r_f = numFeatures - num_features_main;
    //handle the remainder....It is important...
    if (num_r_f > 0)
    {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      for (int k = 0; k < num_r_f; k++)
      {
        tmp_buffer[k] = src[num_features_main + k]; //j is the existing index...
        //printf("tmp_buffer[%d] = 0x%8x\t", k, tmp_buffer[k]);
      }
      //printf("\n");
      for (int k = 0; k < 32; k++) //64 bits for each bit...
      {
        //printf("The %d-th bit:\n", k);

        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
        for (int m = 0; m < 16; m++)
          result_buffer[m] = 0;

        for (int m = 0; m < num_r_f; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }

        //printf("Dest address_index: %d\n", address_index);
        //for (int m = 0; m < 16; m++)
        //  printf("result_buffer[%d]=0x%8x  ", m, result_buffer[m]);

        //printf("\n");

        //if (address_index == 1904)
        //  printf("Address 1904 is here...\n");

          //1--64 
          dest[address_index+0] = result_buffer[0]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+1] = result_buffer[1]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          address_index        += 2;

        if (num_r_f > 64)
        { //65--128 
          dest[address_index+0] = result_buffer[2]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+1] = result_buffer[3]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          address_index        += 2;
        }

        if (num_r_f > 128)
        { //129--256 
          dest[address_index+0] = result_buffer[4]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+1] = result_buffer[5]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+2] = result_buffer[6]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+3] = result_buffer[7]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          address_index        += 4;
        }

        if (num_r_f > 256)
        { //257-511
          dest[address_index+0] = result_buffer[8]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+1] = result_buffer[9]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+2] = result_buffer[10];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+3] = result_buffer[11];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+4] = result_buffer[12];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+5] = result_buffer[13];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+6] = result_buffer[14];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          dest[address_index+7] = result_buffer[15];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
          address_index        += 8;
        }
        
      }             
    }
}


uint32_t regroup_from_bitweaving(uint32_t *dest_addr, uint32_t i, uint32_t numFeatures)
{
  //Validation check...
  if (i >= numFeatures)
  {
    printf("the interested feature with index: %d exceeds the boundary: %d.\n", i, numFeatures);
    return 0;
  }

  //Compute the main part of numFeatures.
  uint32_t num_features_main = (numFeatures/BITS_OF_ONE_CACHE_LINE) * BITS_OF_ONE_CACHE_LINE;

  if (i < num_features_main)
  {
    uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE     ) * BITS_OF_ONE_CACHE_LINE; //
    uint32_t int_offset  = ( i&(BITS_OF_ONE_CACHE_LINE-1) )/32;
    uint32_t bit_offset  = i & 31;

    //The next 32 CLs contains the information of the feature. 
    uint32_t result = 0;
    uint32_t tmp;
    for (uint32_t j = 0; j < 32; j++)
    {
                            //main        bit    which ints 
      tmp     = dest_addr[main_offset + 16 * j + int_offset]; 
      result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (31-j)); //
    }
    return result;
  }
  else
  {
    uint32_t num_r_f = numFeatures - num_features_main;

    if (num_r_f <= 64)                                               //////remainder <= 64
    { 
      uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
      uint32_t int_offset  = ( i & (64-1) )/32;
      uint32_t bit_offset  = i & 31;

      //The next 32 CLs contains the information of the feature. 
      uint32_t result = 0;
      uint32_t tmp;
      for (uint32_t j = 0; j < 32; j++)
      {
                          //main          bit    which ints 
        tmp     = dest_addr[main_offset + 2 * j + int_offset]; 
        result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (31-j)); //
      }
      return result;
    }
    else if (num_r_f <= 128)                                          //////64 < remainder <= 128
    { 
      uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
      uint32_t int_offset  = ( i&(128-1) )/32;
      uint32_t bit_offset  = i & 31;

      //The next 32 CLs contains the information of the feature. 
      uint32_t result = 0;
      uint32_t tmp;
      for (uint32_t j = 0; j < 32; j++)
      {
                          //main          bit    which ints 
        tmp     = dest_addr[main_offset + 4 * j + int_offset]; 
        result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (31-j)); //
      }
      return result;
    }
    else if (num_r_f <= 256)                                          //////128 < remainder <= 256
    { 
      uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
      uint32_t int_offset  = ( i&(256-1) )/32;
      uint32_t bit_offset  = i & 31;

      //The next 32 CLs contains the information of the feature. 
      uint32_t result = 0;
      uint32_t tmp;
      for (uint32_t j = 0; j < 32; j++)
      {
                          //main          bit    which ints 
        tmp     = dest_addr[main_offset + 8 * j + int_offset]; 
        result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (31-j)); //
      }
      return result;
    }
    else if (num_r_f < 512)                                          //////256 < remainder < 512
    { 
      uint32_t main_offset = ( i/BITS_OF_ONE_CACHE_LINE ) * BITS_OF_ONE_CACHE_LINE;
      uint32_t int_offset  = ( i&(512-1) )/32;
      uint32_t bit_offset  = i & 31;
/*        if (i == 0x600)
        {
          printf("main_offset = %d\n", main_offset);
          printf("int_offset  = %d\n", int_offset);
          printf("bit_offset = %d\n", bit_offset);
        }
*/      //The next 32 CLs contains the information of the feature. 
      uint32_t result = 0;
      uint32_t tmp;
      for (uint32_t j = 0; j < 32; j++)
      {
                          //main          bit    which ints 
        tmp     = dest_addr[main_offset + 16 * j + int_offset]; 
        //if (i == 0x600)
        //  printf("%d_%d: 0x%8x,  ", j, main_offset + 16 * j + int_offset, tmp);
        result |= (( (tmp&(1<<bit_offset)) >> bit_offset ) << (31-j)); //
      }
      return result;
    }

    printf("For the remainder, it is not supported yet.\n");
    return 0xF0F0FF00;
        
  }
  //Compute the main offset
}

void test_bitweaving()
{
  //1024+512+257: For this case, the result is wrong!!!!!!!!!!!!!
  uint32_t numFeatures = 1024+512+257; //47236: the case I use is right...
  uint32_t *src_addr, *dest_addr;
  uint32_t numFeatures_real = compute_num_CLs_per_sample(numFeatures);
  //printf("");
 
  srand (time(NULL));  

  src_addr  = (uint32_t *)aligned_alloc(64, numFeatures_real*16 * sizeof(uint32_t));
  dest_addr = (uint32_t *)aligned_alloc(64, numFeatures_real*16 * sizeof(uint32_t));
  
  //1, Initialization... 
  for (uint32_t i = 0; i < numFeatures; i++)
  {
    src_addr[i] = rand();
  }
  printf("Perform bitweaving on the sample with 0x%x features\n", numFeatures);

  //2, Do the bitweaving on it. 
  bitweaving_on_each_sample(dest_addr, src_addr, numFeatures); 
  printf("after bitweaving\n");

  //3, Test the BitWeaving...
  for (uint32_t i = 0; i < numFeatures; i++)
  {
    uint32_t test_result = regroup_from_bitweaving(dest_addr, i, numFeatures);
    if (test_result != src_addr[i])
    {
      printf("\nDamn, there is a bug!!!  Index: 0x%x: original: 0x%x, retrive: 0x%x\n", i, src_addr[i], test_result);
      return;
    }
  }

  printf("Congraturation!!! The test on the 0x%x features passes.\n", numFeatures);

}





void test_srv ()
{
    printf("===============================================================\n");
    printf("= Test __m256i  _mm256_srav_epi32  (__m256i a, __m256i count)  \n");
    printf("===============================================================\n");
    unsigned int data_1 = 0x00000000;
    unsigned int data_2 = 0x11111111;
    unsigned int data_3 = 0x22222222;
    unsigned int data_4 = 0x33333333;
    unsigned int data_5 = 0x44444444;
    unsigned int data_6 = 0x55555555;
    unsigned int data_7 = 0x66666666;
    unsigned int data_8 = 0x77777777;

    __m256i v_offset = _mm256_set_epi32 (7, 6, 5, 4, 3, 2, 1, 0); 
    __m256i v_mask   = _mm256_set1_epi32(0x01010101);

    __m256i v_sum, v_data_1, v_data_2, v_data_3, v_data_4;

    __m256i v_data =  _mm256_set1_epi32 (data_1); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256( v_data, v_mask); //1  v_data
          v_sum    =  v_data_1;

          v_data   =  _mm256_set1_epi32(data_2); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //2  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 1) );

          v_data   =  _mm256_set1_epi32(data_3); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 2) );


          v_data   =  _mm256_set1_epi32(data_4); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 3) );



          v_data   =  _mm256_set1_epi32(data_5); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 4) );



          v_data   =  _mm256_set1_epi32(data_6); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 5) );



          v_data   =  _mm256_set1_epi32(data_7); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 6) );



          v_data   =  _mm256_set1_epi32(data_8); 
          v_data   =  _mm256_srav_epi32(v_data, v_offset); //shift it...
          v_data_1 =  _mm256_and_si256 (v_data, v_mask  ); //3  v_data
          v_sum    =  _mm256_or_si256  (v_sum, _mm256_slli_epi32(v_data_1, 7) );







    int sum_array[8];
    _mm256_store_si256((__m256i *)sum_array, v_sum);


    for (int i = 0; i < 8; i += 1) {
        printf("0x%x ", sum_array[i]);
    }
    printf("\n");
    printf("\n");
}



void test_pow_of_2()
{
    size_t num_threads_2_pow  = 1;

   for (unsigned num_threads = 6; num_threads <= 8; num_threads++)
   {
        for (unsigned base = 64; base > 0; base = (base >> 1) )
        {
          
          if ((num_threads & base) != 0)
          { 
            printf("base = %d\t,", base);
            num_threads_2_pow = base;          //leading one bit
            if ( (num_threads & (base-1)) != 0 ) //handling the remainder... if not zero, then to next value...
            {
              num_threads_2_pow =  num_threads_2_pow << 1;
            }
            break;
          }
        }
        printf("num_threads: %d, power_of_2: %d\n", num_threads, num_threads_2_pow);
    }

}        

void test_blend ()
{
    printf("===============================================================\n");
    printf("= Test __m256i  _mm256_mask_blend_epi32 (__mmask8 k, __m256i a, __m256i b) \n");
    printf("===============================================================\n");
    int imm8_1 = 0x00;
    int imm8_2 = 0x11;
    int imm8_3 = 0x22;
    int imm8_4 = 0x33;
    int imm8_5 = 0x44;
    int imm8_6 = 0x55;
    int imm8_7 = 0x66;
    int imm8_8 = 0x77;

   __m256i v_8_ff = _mm256_set1_epi32 (0x80);
   __m256i v_7_ff = _mm256_set1_epi32 (0x40);
   __m256i v_6_ff = _mm256_set1_epi32 (0x20);
   __m256i v_5_ff = _mm256_set1_epi32 (0x10);
   __m256i v_4_ff = _mm256_set1_epi32 (0x8);
   __m256i v_3_ff = _mm256_set1_epi32 (0x4);
   __m256i v_2_ff = _mm256_set1_epi32 (0x2);
   __m256i v_1_ff = _mm256_set1_epi32 (0x1);
   __m256i v_00   = _mm256_set1_epi32 (0x0);

    __m256i v_sum =                        _mm256_blend_epi32 (v_00, v_8_ff, imm8_8) ; //8-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_7_ff, imm8_7)); //7-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_6_ff, imm8_6)); //6-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_5_ff, imm8_5)); //5-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_4_ff, imm8_4)); //4-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_3_ff, imm8_3)); //3-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_2_ff, imm8_2)); //2-bit
	    v_sum = _mm256_or_si256(v_sum, _mm256_blend_epi32 (v_00, v_1_ff, imm8_1)); //1-bit


	int sum_array[8];
	_mm256_store_si256((__m256i *)sum_array, v_sum);


    for (int i = 0; i < 8; i += 1) {
        printf("%x ", sum_array[i]);
    }
    printf("\n");
    printf("\n");
}



void test_permute ()
{
    printf("===============================================================\n");
    printf("= Test _mm256_permute_ps\n");
    printf("===============================================================\n");

    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    for (int i = 0; i < 8; i += 1) {
        printf("%f ", x[i]);
    }
    printf("\n");
    __m256 tmp = _mm256_loadu_ps(x);
    tmp = _mm256_permute_ps(tmp, 245);
    _mm256_storeu_ps(x, tmp);
    for (int i = 0; i < 8; i += 1) {
        printf("%f ", x[i]);
    }
    printf("\n");
    printf("\n");
}


void test_norm ()
{

    printf("===============================================================\n");
    printf("= Test quantize_get_norm\n");
    printf("===============================================================\n");


    float z[127];

    for (int i = 0; i < 127; i += 1) {
        z[i] = i;
        if (i % 2 == 0) {
            z[i] = z[i] * (-1);
        }
    }

    for (int i = 0; i < 127; i += 1) {
        printf("%f ", z[i]);
    }
    printf("\n");

   // float t = quantize_get_norm(z, 127);
    //printf("norm is: %f\n", t);
    printf("\n");
}


void main ()
{
  test_bitweaving();
  //test_num_CLs();
  //test_pow_of_2();
  //test_srv();
  //test_blend();
  //test_permute();
  //test_norm();
}
