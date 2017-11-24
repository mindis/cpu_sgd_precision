#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>       /* time */
#ifdef __linux__
    #include <malloc.h>
#endif

#include "../BitWeaving/Bitweaving.h"
#include "hazy/vector/fvector.h"


#include "hazy/scan/binfscan.h"
#include "hazy/scan/tsvfscan.h"

//The related parameters are here. 
#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

#define DIMENSION   47236 //512 *3//
#define NUM_SAMPLES 20242 //512 *3//

//Parameter 1 (dimension):   dimension of the training dataset
//Parameter 2 (num_samples): number of samples in the training dataset
//Parameter 3: format of the input dataset, 0: Binary, 1, Matlab_TSV, 2, TSV...
//Parameter 4: Name of input compressed training dataset.
//Parameter 5: Name of output training dataset, which is mmaped to the memory.

void main (int argc, char **argv)
{
  assert(argc > 6);   //At least has the dimentsion and samples.
    //////////////////Input parameters//////////////////
  int    dimension       = argc > 1 ? atoi(argv[1]) : DIMENSION  ;
  size_t num_samples     = argc > 2 ? atoi(argv[2]) : NUM_SAMPLES;
  int    model           = argc > 3 ? atoi(argv[3]) : 2;
  int    num_bits_per_mr = argc > 4 ? atoi(argv[4]) : 2;
  char  *input_filename  = (char *)argv[5];
  char  *output_filename = (char *)argv[6]; // "../../../data/data_16G_4k.dat"

  //argv[3]: name of file to store the dense dataset...

  printf("Samples: %d, dimension: %d, model: %d. \n",num_samples, dimension, model);
  printf("Input file name: %s, output file name: %s. \n",input_filename, output_filename);

  BitWeavingBase bw_master(output_filename, dimension, num_bits_per_mr, num_samples, false);
  bw_master.statistic_show(); 

  if (model == 0) {
    hazy::scan::BinaryFileScanner scan(input_filename);
    bw_master.write_file_to_bitweaving(scan, dimension, num_samples);
    //Loader::LoadSamples(scan, train_samps, dimension);
  } else if (model == 1) {
    hazy::scan::MatlabTSVFileScanner scan(input_filename);
    bw_master.write_file_to_bitweaving(scan, dimension, num_samples);
    //Loader::LoadSamples(scan, train_samps, dimension);
  } else if (model == 2) {
    hazy::scan::TSVFileScanner scan(input_filename);
    bw_master.write_file_to_bitweaving(scan, dimension, num_samples);
    //Loader::LoadSamples(scan, train_samps, dimension);
  } else {
    printf("The model: %d is not supported yet.\n", model);
    return;
  }

/*
  for (int samp_index = 0; samp_index < num_samples; samp_index++)
  {
    unsigned int *data = (unsigned int *)malloc(dimension*sizeof(unsigned int)); //[NUM_VALUES];
    for (int i = 0; i < dimension; i++)
      data[i] = rand(); //(float)i;
 
    hazy::vector::FVector<unsigned int> samp(data, dimension);

    bw_master.write_to_bitweaving(samp_index, samp, 0.1); //Write one sample to the BitWeaving memory space!!!!!!

    LinearModelSampleBitweaving samp_bitweaving;

    bw_master.read_from_bitweaving(samp_index, samp_bitweaving);

  //Set up the destination...
    unsigned short dest[2*dimension];
    hazy::vector::FVector<unsigned short> dest_short_vector (dest, dimension);

    samp_bitweaving.Unpack_from_bitweaving(dest_short_vector, 9);

    for (int i = 0; i < dimension; i++)
      if ( ( (data[i]>>23)<<7 ) != dest_short_vector[i])
      {
        printf("Error::::::%d: src_0x%8x, dest_0x%x\n", i, data[i], dest_short_vector[i]);
        return;
      }
  }
  
  printf("Congratuation!!! Your test is passed...\n");   
*/
}
