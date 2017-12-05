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


#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <time.h>       /* time */
#ifdef __linux__
    #include <malloc.h>
#endif


#include "global_macros.h"			  //Configurations...
#include "frontend_util.h"            //Handle the input arguments...
#include "linearmodel/linearmodel.h"  //Parameter of linear model...
#include "BitWeaving/Bitweaving.h"    //Sample of linear model....

#include "linearmodel/types/thread_args.h"  //Parameter of linear model...


#if defined(_HOGWILD) 
	#include "strategy/hogwild.h"
#elif defined(_MODELSYNC)
	#include "strategy/modelsync.h"
#else

#endif



#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#ifdef CPU_BINDING_EN
#include "hazy/thread/thread_pool-inl_binding.h"
#else
#include "hazy/thread/thread_pool-inl.h"
#endif


//#include "linearmodel_exec.h"
//#include "linreg/linreg_exec.h"


template< class Sample > 
void b_normalize(Sample *samps, uint32_t num_samps, uint32_t target_label, uint32_t class_model, uint32_t num_bits)
{
  bool normalize_enable   = class_model&1;     //first bit..
  bool toMinus1_1         = (class_model>>1)&1;//second bit..
  float base              = ( (num_bits<=8)? 256.0 : 65536.0 );//samps[0].b_binary_to_value();
  float not_targeted      = toMinus1_1? (base*(-1.0)): 0.0;
  //unsigned num_samps      = samps.size;
  
  printf("base = %f, class_model = %d, normalize_enable = %d, toMinus1_1=%d, num_samps= %d\n", base, class_model, normalize_enable, toMinus1_1, num_samps );
    
  if (normalize_enable)
  {
    for (int i = 0; i < num_samps; i++)
    {
      if (samps[i].rating != target_label) //not the target label, assign to -1 or 0. 
        samps[i].rating = not_targeted;
      else
        samps[i].rating = base;//1.0;             //1.0 for the targeted label...
    }
  }
  return;
}


int main(int argc, char** argv)
{
#if defined(_HOGWILD) 
    Hogwild<LinearModel, LinearModelParams, LinearModelSampleBitweaving> executor;
  	printf("Use Hogwild.......\n");
#elif defined(_MODELSYNC)
    ModelSync<LinearModel, LinearModelParams, LinearModelSampleBitweaving> executor;
  	printf("Use ModelSync.......\n");
#else
	printf("Wrong execution strategy, not modelsync or hogwild!!!\n");    	
#endif

#ifdef AVX2_EN
    printf("Using AVX!.......\n");  
#else
    printf("Using SCALAR!.......\n");  

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////	
//////////////////////////////Handle the parameters...///////////////////////////////////////////////////////	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	unsigned nepochs = 20;
	unsigned nthreads = 1;
	uint32_t dimension = -1;
	float step_size = 5e-2, step_decay = 0.8, beta = 0.55;
	float batch_size = 0; // represents no batch
	float lasso_regularizer = 0; // Says no regularizer
	unsigned quantization = 0; // No quantization
	unsigned quantizationLevel = 0;
	unsigned class_model  = 0;
	unsigned target_label = 1;
	unsigned target_epoch = 0;
	unsigned num_bits     = 8;
	unsigned model        = 0; //0: RCV1,
  unsigned bits_per_mr  = 4;
  unsigned huge_page_en = 0;

  static struct extended_option long_options[] = {
    {"beta", required_argument, NULL,              'h', "the exponent constant for the stepsizes"},
    {"batch_size", required_argument, NULL,        'b', "batch_size (default to 1)"},
    {"dimension"    ,required_argument, NULL,      'd', "dimension"},
    {"epochs"    ,required_argument, NULL,         'e', "number of epochs (default is 20)"},
    {"stepinitial",required_argument, NULL,        'i', "intial stepsize (default is 5e-2)"},
    {"step_decay",required_argument, NULL,         'x', "stepsize decay per epoch (default is 0.8)"},
    {"lasso_regularizer", required_argument, NULL, 'a', "lasso regularizer (L1 norm)"},
    {"quantization", required_argument, NULL,      'q', "quantization (for LinReg) 0: No quantization / 1: Quatize samples / 2: Quantize gradients / 3: Quantize samples & gradient / 4: Quantize model / 5: Quantize model & samples / 6: Quantize model & gradient / 7: Quantize model & samples & gradient"},
    {"qlevel", required_argument, NULL,            'l', "Quantization level"},
    {"splits", required_argument, NULL,            'r', "number of thread per working process (default is 1)"},
    //"binary", required_argument,NULL,             'v', "load the file in a binary fashion"},
    //{"matlab-tsv", required_argument,NULL,         'm', "load TSVs indexing from 1 instead of 0"},
    {"model", required_argument,NULL,              'm', "The index of the sample..."},
    {"bits_per_mr",     required_argument,NULL,    'g', "Number of bits for each memory region. default:4"},
    {"huge_page_en",     required_argument,NULL,   'z', "Using huge page or not. default:false"},

    {"class_model",  required_argument,NULL,       'c', "First bit: enable binary classification, second bit: -1 or 0  Default: 0"},
    {"target_label", required_argument,NULL,       't', "Target label to be identified. default:1"},
    {"pcm_epoch", required_argument,NULL,          'p', "Target epoch to be analyzed. Default:0 "},
    {"num_bits",     required_argument,NULL,       'n', "Number of bits used. default:8"},
	  
    {NULL,0,NULL,0,0}
  };

    char usage_str[] = "<train file> <test file> <metadata file>";
    int c = 0, option_index = 0;
    option* opt_struct = convert_extended_options(long_options);
    while( (c = getopt_long(argc, argv, "", opt_struct, &option_index)) != -1) 
    {
      switch (c) { 
        case 'h':
          beta = atof(optarg);
          break;
        case 'e':
          nepochs = atoi(optarg);
          break;
        case 'b':
          batch_size = atoi(optarg);
          break;
        case 'a':
          lasso_regularizer = atof(optarg);
          break;
        case 'i':
          step_size = atof(optarg);
          break;
        case 'x':
          step_decay = atof(optarg);
          break;
        case 'd':
          dimension = atoi(optarg);
          break;
        case 'r':
          nthreads = atoi(optarg);
          break;
        case 'q':
          quantization = atoi(optarg);
          break;
        case 'l':
          quantizationLevel = atoi(optarg);
          break;
        case 'c':
          class_model       = atoi(optarg);
          break;	
        case 't':
          target_label      = atoi(optarg);
        case 'n':
          num_bits          = atoi(optarg);		  
          break;			  
        case 'p':
          target_epoch	    = atoi(optarg);
          break;				
        case 'm':
          model             = atoi(optarg);
          break;
        case 'g':
          bits_per_mr       = atoi(optarg);
          break;
        case 'z':
          huge_page_en      = atoi(optarg);
          break;

        case ':':
        case '?':
		  printf("wrong parameter: %d\n", c);
          print_usage(long_options, argv[0], usage_str);
          exit(-1);
          break;
      }
    }

    char *szTrainFile, *szTestFile;

    if(optind == argc - 2) {
      szTrainFile = argv[optind];
      szTestFile  = argv[optind + 1];
    } else {
      printf("wrong training_test dataset format\n");
      print_usage(long_options, argv[0], usage_str);
      exit(-1);
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////	
//////////////////////////////Try to read the train dataset//////////////////////////////////////////////////	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////	
    uint32_t num_samples;
    if (model == 0)
    {
    	dimension   = 47236;
    	num_samples = 20242;
    }
    else
    {
    	dimension   = 47236;
    	num_samples = 20242;
    }


    LinearModelParams p (step_size, step_decay);
    p.batch_size        = batch_size;
    p.beta              = beta;
    p.lasso_regularizer = lasso_regularizer;
    p.quantization      = quantization;
    p.quantizationLevel = quantizationLevel;

    p.class_model       = class_model;
    p.target_label      = target_label;
    p.target_epoch      = target_epoch;
    p.num_bits          = num_bits;	
    p.model             = model;
    p.num_samples       = num_samples;
    p.ndim              = dimension;
	p.nthreads			= nthreads;
	
	//Add two parameters here. Number of bits for each region, and huge table enable..
	BitWeavingBase bw_master(szTrainFile, dimension, bits_per_mr, num_samples, huge_page_en, true); //false
	///////////////Add the other file when necessary...///////////////////

	printf("step 1: prepare the training dataset. dimension = %d, num_samples = %d\n", dimension, num_samples);
	//////////Malloc the space for the dataset/////////
	LinearModelSampleBitweaving* p_samp = (LinearModelSampleBitweaving*)malloc(num_samples*sizeof(LinearModelSampleBitweaving));
	for (uint32_t samp_index = 0; samp_index < num_samples; samp_index++)
	{
		bw_master.read_from_bitweaving(samp_index,  p_samp[samp_index]);
	}
	/////////////////p_samp only has the address..///////////////////////

	printf("step 2: Normalize the rating.\n");
	b_normalize(p_samp, num_samples, target_label, class_model, num_bits);


	/////////////////Initialization of Linear Model///////////////////////
	LinearModel *model_ = new LinearModel(dimension, nthreads);


	//d

	//args.losses_        = this->losses_;
	//args.compute_times_ = this->compute_times_ ;
 	executor.Run(model_, p, p_samp, nepochs);

 	delete model_;
	free(p_samp);

}
