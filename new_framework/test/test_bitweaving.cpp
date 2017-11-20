#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>       /* time */
#ifdef __linux__
    #include <malloc.h>
#endif

//#include "sample.h"
#include "Bitweaving.h"
#include "hazy/vector/fvector.h"


//The related parameters are here. 
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

#define NUM_VALUES 47236 //512 *3//
#define NUM_SAMPLES 23102 //512 *3//


void test_bitweaving()
{  //23000

  BitWeavingBase bw_master("../../../data/data_16G_4k.dat", NUM_VALUES, 4, NUM_SAMPLES, false); //(const char *fname, uint32_t dimension, uint32_t num_bits, uint32_t num_samples )
  //BitWeavingBase test(47236, 4, 23000, false);
  bw_master.statistic_show(); 

for (int samp_index = 0; samp_index < NUM_SAMPLES; samp_index++)
{
   unsigned int *data = (unsigned int *)malloc(NUM_VALUES*sizeof(unsigned int)); //[NUM_VALUES];
  for (int i = 0; i < NUM_VALUES; i++)
    data[i] = rand(); //(float)i;
 
  hazy::vector::FVector<unsigned int> samp(data, NUM_VALUES);

  bw_master.write_to_bitweaving(samp_index, samp, 0.1); //Write one sample to the BitWeaving memory space!!!!!!

  LinearModelSampleBitweaving samp_bitweaving;

  bw_master.read_from_bitweaving(samp_index, samp_bitweaving);
/*
  printf("rating = %f\n",      samp_bitweaving.rating);
  printf("dimension   = %d\n", samp_bitweaving.dimension);
  printf("align_bits  = %d\n", samp_bitweaving.align_bits);
  printf("num_regions = %d\n", samp_bitweaving.num_regions);
  printf("region_offset = 0x%lx\n", samp_bitweaving.region_offset);
*/
  //samp_bitweaving.print_info();


#if 0

  //Set up the destination...
  unsigned char dest[2*NUM_VALUES];
  hazy::vector::FVector<unsigned char> dest_char_vector (dest, NUM_VALUES);

  samp_bitweaving.Unpack_from_bitweaving(dest_char_vector, 8);

  for (int i = 0; i < NUM_VALUES; i++)
    if ( ( (data[i]>>24)<<0 ) != dest_char_vector[i])
    {
      printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_char_vector[i]);
      return;
    }
}
#else
    //Set up the destination...
    unsigned short dest[2*NUM_VALUES];
    hazy::vector::FVector<unsigned short> dest_short_vector (dest, NUM_VALUES);

    samp_bitweaving.Unpack_from_bitweaving(dest_short_vector, 9);

    for (int i = 0; i < NUM_VALUES; i++)
        if ( ( (data[i]>>23)<<7 ) != dest_short_vector[i])
        {
            printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_short_vector[i]);
            return;
        }
}
#endif

        //if (data[i] != 0.0)
        //{
        //    printf("ERROR: %d, %f\n", i, data[i]);
        //    break;
        //}
    printf("Congratuation!!! Your test is passed...\n");   



}


 
/*
void test_short_Convert_from_bitweaving()
{
    printf("===============================================================\n");
    printf("================ Test Convert_from_bitweaving on short=================\n");
    printf("===============================================================\n");

    srand (time(NULL));  

    //Set up the source vector with unsigned int....
    unsigned int *data = (unsigned int *)malloc(2*NUM_VALUES*sizeof(unsigned int)); //[NUM_VALUES];
    for (int i = 0; i < NUM_VALUES; i++)
        data[i] = rand(); //(float)i;
 
    //Store the compressed data...
    unsigned int data_bitweaving[NUM_VALUES*2];
    hazy::vector::bitweaving_on_each_sample(data_bitweaving, data, NUM_VALUES);

   
    hazy::vector::FVector<unsigned int> src_int_vector (data_bitweaving, NUM_VALUES);


    //Set up the destination...
    unsigned short dest[2*NUM_VALUES];

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
*/


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
