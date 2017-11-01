#ifndef _LINEARMODEL_EXEC_H
#define _LINEARMODEL_EXEC_H

#include <stdlib.h>

#include "global_macros.h"

#include "linearmodel/linearmodel.h"
#include "linearmodel/linearmodel_loader.h"

#include "frontend_util.h"

#if defined(_ALLREDUCE)
#include "strategy/all_reduce_sync.h"
#elif defined(_ALLREDUCESPARSE)
#include "strategy/all_reduce_sparse.h"
#elif defined(_ALLREDUCEBINOMIAL)
#include "strategy/all_reduce_binomial_sync.h"
#elif defined(_ASYNC)
#include "strategy/parameter_server_async.h"
#elif defined(_ASYNCSPARSE)
#include "strategy/parameter_server_async_sparse.h"
#elif defined(_SYNC)
#include "strategy/parameter_server_sync.h"
#else
#include "strategy/hogwild.h"
#endif

class LinearModelExec
{
public:
  template< class Exec > static int Run(int argc, char** argv)
  {
#if defined(_ALLREDUCE)
    AllReduceSync<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#elif defined(_ALLREDUCESPARSE)
    AllReduceSparse<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#elif defined(_ALLREDUCEBINOMIAL)
    AllReduceBinomialSync<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#elif defined(_ASYNC)
    ParameterServerAsync<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#elif defined(_ASYNCSPARSE)
    ParameterServerAsyncSparse<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#elif defined(_SYNC)
    ParameterServerSync<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
#else 
  #if defined(_HOGWILD_SHORT) 
    Hogwild<LinearModel, LinearModelParams, LinearModelSample_short, LinearModelLoader, Exec> executor;
  	printf("Use short.......\n");
  #elif defined(_HOGWILD_CHAR) 
	  Hogwild<LinearModel, LinearModelParams, LinearModelSample_char, LinearModelLoader, Exec> executor;
	  printf("Use char.......\n");
  #elif defined(_HOGWILD_INT) 
	  Hogwild<LinearModel, LinearModelParams, LinearModelSample_int, LinearModelLoader, Exec> executor;
	  printf("Use int: BitWeaving.......\n");	  
  #else 
    Hogwild<LinearModel, LinearModelParams, LinearModelSample, LinearModelLoader, Exec> executor;
  	printf("Use floating point.......\n");  
  #endif

  #ifdef AVX2_EN
    printf("Using AVX!.......\n");  
  #endif
#endif

    bool matlab_tsv = false;
    bool loadBinary = false;
    unsigned nepochs = 20;
    unsigned nthreads = 1;
    int dimension = -1;
    float step_size = 5e-2, step_decay = 0.8, beta = 0.55;
    float batch_size = 0; // represents no batch
    float lasso_regularizer = 0; // Says no regularizer
    unsigned quantization = 0; // No quantization
    unsigned quantizationLevel = 0;
	unsigned class_model  = 0;
	unsigned target_label = 1;
	unsigned target_epoch = 0;
	unsigned num_bits = 8;

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
      {"binary", required_argument,NULL,             'v', "load the file in a binary fashion"},
      {"matlab-tsv", required_argument,NULL,         'm', "load TSVs indexing from 1 instead of 0"},

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
        case 'v':
          loadBinary = (atoi(optarg) != 0);
          break;
        case 'm':
          matlab_tsv = (atoi(optarg) != 0);
          break;
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
          target_epoch	     = atoi(optarg);
          break;				
		  
        case ':':
        case '?':
		  printf("wrong parameter: %d\n", c);
          print_usage(long_options, argv[0], usage_str);
          exit(-1);
          break;
      }
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

    char *szTrainFile, *szTestFile, *szMetadataFile;
	szMetadataFile = NULL;

    if(optind == argc - 2) {
      szTrainFile = argv[optind];
      szTestFile  = argv[optind + 1];
    } else {
      printf("wrong training_test dataset format\n");
      print_usage(long_options, argv[0], usage_str);
      exit(-1);
    }

    executor.params_ = &p;
    executor.Init(szTrainFile, szTestFile, szMetadataFile, loadBinary, matlab_tsv, dimension, nthreads);

    executor.Run(nepochs);
    return 0;
  }
};

#endif
