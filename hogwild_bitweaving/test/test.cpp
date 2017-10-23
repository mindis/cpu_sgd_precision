#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef __linux__
    #include <malloc.h>
#endif


void test_srv ()
{
    printf("===============================================================\n");
    printf("= Test __m256i  _mm256_srav_epi32  (__m256i a, __m256i count) \n");
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
    test_pow_of_2();

    test_srv();
    //test_blend();
    //test_permute();
    //test_norm();

}
