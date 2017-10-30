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


void test_Convert_from_bitweaving()
{
    printf("===============================================================\n");
    printf("================ Test Convert_from_bitweaving =================\n");
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
    unsigned char dest[NUM_VALUES];

    hazy::vector::FVector<unsigned char> dest_char_vector (dest, NUM_VALUES);

    hazy::vector::Convert_from_bitweaving(dest_char_vector, src_int_vector, 8);

    //    sample_char.regroup_from_bitweaving(samps[i], num_bits);


    for (int i = 0; i < NUM_VALUES; i++)
        if ( ( (data[i]>>24)<<0 ) != dest_char_vector[i])
        {
            printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_char_vector[i]);
            return;
        }
        //if (data[i] != 0.0)
        //{
        //    printf("ERROR: %d, %f\n", i, data[i]);
        //    break;
        //}
    printf("Congratuation!!! Your test is passed...");
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
    test_Convert_from_bitweaving(); //test_fvector_zero();
    //test_blend();
    //test_permute();
    //test_norm();

}
