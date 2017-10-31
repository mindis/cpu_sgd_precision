#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef __linux__
    #include <malloc.h>
#endif

#define AVX2_EN



#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#define NUM_VALUES 47236

void test_mm256_shuffle_epi8()
{
    printf("===============================================================\n");
    printf("================ Test _mm256_shuffle_epi8=================\n");
    printf("===============================================================\n");


    __m256i v_sum              = _mm256_set_epi8 (31,  23,  15,  7, 
                                                  30,  22,  14,  6, 
                                                  29,  21,  13,  5, 
                                                  28,  20,  12,  4,
                                                  27,  19,  11,  3, 
                                                  26,  18,  10,  2, 
                                                  25,  17,   9,  1, 
                                                  24,  16,   8,  0);

    __m256i v_shuffle_constant = _mm256_set_epi8 (15, 11,  7,  3, 
                                                  14, 10,  6,  2, 
                                                  13,  9,  5,  1, 
                                                  12,  8,  4,  0,
                                                  15, 11,  7,  3, 
                                                  14, 10,  6,  2, 
                                                  13,  9,  5,  1, 
                                                  12,  8,  4,  0);

    __m256i v_data_2 = _mm256_shuffle_epi8(v_sum, v_shuffle_constant);


    unsigned char sum_array[64]; //32 is enough.
    _mm256_store_si256((__m256i *)sum_array, v_data_2);
    printf("v_data_2 tmp result:\n");

    for (unsigned k = 0; k < 32; k+=4) //it is possible to use AVX instructions?
        printf("%d %d %d %d\n", sum_array[31-k], sum_array[30-k], sum_array[29-k], sum_array[28-k]);
        //vec_char[base + offset*32 + k] = sum_array[(k>>3)+((k&7)<<2)]; 




    __m256i v_perm_constant = _mm256_set_epi32 (7, 3,  6, 2,   
                                                5,  1, 4,  0); 
    __m256i v_result = _mm256_permutevar8x32_epi32(v_data_2, v_perm_constant);

   // unsigned char sum_array[64]; //32 is enough.
    _mm256_store_si256((__m256i *)sum_array, v_result);
    printf("Final result:\n");

    for (unsigned k = 0; k < 32; k+=4) //it is possible to use AVX instructions?
        printf("%d %d %d %d\n", sum_array[31-k], sum_array[30-k], sum_array[29-k], sum_array[28-k]);
        
}



void test_short_Convert_from_bitweaving()
{
    printf("===============================================================\n");
    printf("================ Test Convert_from_bitweaving on short=================\n");
    printf("===============================================================\n");

    srand (time(NULL));  

    //Set up the source vector with unsigned int....
    unsigned int data[NUM_VALUES];
    for (int i = 0; i < NUM_VALUES; i++)
        data[i] = rand(); //(float)i;

    //Store the compressed data...
    unsigned int data_bitweaving[NUM_VALUES*2];
    hazy::vector::bitweaving_on_each_sample(data_bitweaving, data, NUM_VALUES);


    hazy::vector::FVector<unsigned int> src_int_vector (data_bitweaving, NUM_VALUES);


    //Set up the destination...
    unsigned short dest[NUM_VALUES];

    hazy::vector::FVector<unsigned short> dest_char_vector (dest, NUM_VALUES);

    hazy::vector::Convert_from_bitweaving(dest_char_vector, src_int_vector, 16);

    //    sample_char.regroup_from_bitweaving(samps[i], num_bits);


    for (int i = 0; i < NUM_VALUES; i++)
        if ( ( (data[i]>>16)<<0 ) != dest_char_vector[i])
        {
            printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_char_vector[i]);
            return;
        }
        //if (data[i] != 0.0)
        //{
        //    printf("ERROR: %d, %f\n", i, data[i]);
        //    break;
        //}
    printf("Congratuation!!! Your test is passed...\n");
}        

void test_char_Convert_from_bitweaving()
{
    printf("===============================================================\n");
    printf("============ Test Convert_from_bitweaving to char==============\n");
    printf("===============================================================\n");

    srand (time(NULL));  

    //Set up the source vector with unsigned int....
    unsigned int *data = (unsigned int *)malloc(2*NUM_VALUES*sizeof(unsigned int)); //[NUM_VALUES];
    //unsigned int data[NUM_VALUES];

    for (int i = 0; i < NUM_VALUES; i++)
    {
        data[i] = rand(); //(float)i;
        if (i < 10)
            printf("data[%d] = 0x%8x\n", i, data[i]);
    }
    //Store the compressed data...
    unsigned int data_bitweaving[NUM_VALUES*2];
    hazy::vector::bitweaving_on_each_sample(data_bitweaving, data, NUM_VALUES);


    hazy::vector::FVector<unsigned int> src_int_vector (data_bitweaving, NUM_VALUES);


    //Set up the destination...
    unsigned char dest[NUM_VALUES+512];

    hazy::vector::FVector<unsigned char> dest_char_vector (dest, NUM_VALUES);

    hazy::vector::Convert_from_bitweaving(dest_char_vector, src_int_vector, 8);

    //    sample_char.regroup_from_bitweaving(samps[i], num_bits);


    for (int i = 0; i < 32; i++)
        if ( ( (data[i]>>24) ) != dest_char_vector[i])
        {
            printf("Error::::::%d: src_0x%8x, dest_0x%2x\n", i, data[i], dest_char_vector[i]);
            return;
        }
        //if (data[i] != 0.0)
        //{
        //    printf("ERROR: %d, %f\n", i, data[i]);
        //    break;
        //}
    free(data);

    printf("Congratuation!!! Your test is passed...\n");

    return;
}    

#if 0
void test_fvector_zero()
{
    printf("===============================================================\n");
    printf("================ Test FVECTOR ZERO ==============================.\n");
    printf("===============================================================\n");

    float data[NUM_VALUES];
    for (int i = 0; i < NUM_VALUES; i++)
        data[i] = (float)i;

    hazy::vector::FVector<float> test_vector (data, NUM_VALUES);

    hazy::vector::zero(test_vector);

    for (int i = 0; i < NUM_VALUES; i++)
        if (data[i] != 0.0)
        {
            printf("ERROR: %d, %f\n", i, data[i]);
            break;
        }

    //sf;
}        

#endif


void main ()
{ 
    #ifdef AVX2_EN
        printf("Using AVX2!!!\n");
    #else
        printf("Using Scalar!!!\n");
    #endif
    //test_mm256_shuffle_epi8();
    test_char_Convert_from_bitweaving();
    //test_short_Convert_from_bitweaving();
    //test_Convert_from_bitweaving(); //test_fvector_zero();
    //test_blend();
    //test_permute();
    //test_norm();

}
