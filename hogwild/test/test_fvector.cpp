#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef __linux__
    #include <malloc.h>
#endif

#define AVX2_EN



#ifdef AVX2_EN
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#include "hazy/vector/operations-inl_avx2.h"
#else
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#include "hazy/vector/operations-inl.h"
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

    hazy::vector::zero(test_vector);

    for (int i = 0; i < NUM_VALUES; i++)
        if (data[i] != 0.0)
        {
            printf("ERROR: %d, %f\n", i, data[i]);
            break;
        }

    //sf;
}        



void main ()
{
    test_fvector_zero();
    //test_blend();
    //test_permute();
    //test_norm();

}
