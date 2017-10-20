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

#define NUM_VALUES 111

void test_fvector_zero()
{
    printf("===============================================================\n");
    printf("================ Test FVECTOR ZERO ==============================.\n");
    printf("===============================================================\n");

    float data[NUM_VALUES];
    for (int i = 0; i < NUM_VALUES; i++)
        data[i] = (float)i;

    hazy::vector::FVector<float> test_vector (data, NUM_VALUES);

    hazy::vector::Zero(test_vector);

    for (int i = 0; i < NUM_VALUES; i++)
        if (data[i] != 0.0)
        {
            printf("ERROR: %d, %f\n", i, data[i]);
            break;
        }

    printf("Test result with %d floats is OK!!!!\n", NUM_VALUES);//sf;
}        

void test_fvector_copyto()
{
    printf("===============================================================\n");
    printf("================ Test FVECTOR copyto ==============================.\n");
    printf("===============================================================\n");

    float src[NUM_VALUES], dest[NUM_VALUES];
    for (int i = 0; i < NUM_VALUES; i++)
        src[i] = (float)i;

    for (int i = 0; i < NUM_VALUES; i++)
        dest[i] = (float)(NUM_VALUES-i);



    hazy::vector::FVector<float> src_vector  (src,  NUM_VALUES);
    hazy::vector::FVector<float> dest_vector (dest, NUM_VALUES);

    hazy::vector::CopyInto(src_vector, dest_vector);

    for (int i = 0; i < NUM_VALUES; i++)
        if (dest[i] != (float)i)
        {
            printf("ERROR: %d, %f\n", i, dest[i]);
            break;
        }

    printf("Test result with %d floats is OK!!!!\n", NUM_VALUES);//sf;
}        



void main ()
{
    test_fvector_zero();

    test_fvector_copyto();
    //test_blend();
    //test_permute();
    //test_norm();

}
