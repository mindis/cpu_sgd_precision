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
//#else
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

#include "perf_counters.h"
struct Monitor_Event inst_Monitor_Event = {
	{
		{0x2e,0x41},
		{0x24,0x21},
		{0xc5,0x00},
		{0x24,0x41},
	},
	1,
	{
		"L3 cache misses: ",
		"L2 cache misses: ",
		"Mispredicted branchs: ",
		"L2 cache hits: ",
	},
	{
		{0,0},
		{0,0},
		{0,0},
		{0,0},		
	},
	2,
	{
		"MIC_0",
		"MIC_1",
		"MIC_2",
		"MIC_3",
	},
    0	 
};








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



template< class Model, class Params, class Sample> 
void ComputeLossPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
{
	Model *model         =  threadArgs.model_ ;
    Params const &params = *threadArgs.params_;
    Sample     *samps    =  threadArgs.samples_;
    double    *loss_addr =  threadArgs.losses_;

	unsigned num_bits	 =  params.num_bits;
	size_t batch_size	 =  params.batch_size;
	uint32_t num_samples =  params.num_samples;
	uint32_t dimension   =  params.ndim;

    hazy::vector::FVector<fp_type> &x       = model->weights;


    // calculate which chunk of examples we work on
    size_t start = (num_samples / total) * tid; //GetStartIndex(sampsvec.size, tid, total); 
    size_t end   = (total == tid+1)? num_samples : (start + num_samples/total);//GetEndIndex(sampsvec.size, tid, total);

    double sum_loss = 0.0;

    if (num_bits <= 8)
    {

	   unsigned char dest[512+dimension];
	   hazy::vector::FVector<unsigned char> dest_char_vector (dest, dimension);

      for (unsigned i = start; i < end; i++) 
      {
		//2.1: Regroup the data from bitweaving...
		samps[i].Unpack_from_bitweaving(dest_char_vector, num_bits); //hazy::vector::Convert_from_bitweaving(dest_char_vector, samps[i].vector, num_bits);

		//2.2: Compute the loss value...
        fp_type delta;
        delta =  (Dot( x, dest_char_vector) - samps[i].rating)/256.0; //0.01;//

		sum_loss += 0.5*delta*delta;       
      }  
    }
    else if (num_bits <= 16)  
    {

		unsigned short dest[512+dimension];
		hazy::vector::FVector<unsigned short>dest_short_vector(dest, dimension);
		
		for (unsigned i = start; i < end; i++) 
		{
			//2.1: Regroup the data from bitweaving...
			samps[i].Unpack_from_bitweaving(dest_short_vector, num_bits); //hazy::vector::Convert_from_bitweaving(dest_char_vector, samps[i].vector, num_bits);
			  
			//2.2: Compute the loss value...
			fp_type delta;
			delta = (Dot( x, dest_short_vector) - samps[i].rating)/65536.0;
			sum_loss += 0.5*delta*delta;       
			  	  
		}  
	}
	else if (num_bits <= 32)
    {
      printf("Bits: %d. Not Supported yet...", num_bits);
    }

    loss_addr[tid] = sum_loss;

	//threadArgs.compute_times_[tid].ptr->Pause();//wzk: pure computation time...

}




template< class Model, class Params, class Sample> 
void ModelSyncPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
{
	Model *model         =  threadArgs.model_ ;
    Params const &params = *threadArgs.params_;
    Sample     *samps    =  threadArgs.samples_;


	unsigned num_bits	 =  params.num_bits;
	size_t batch_size	 =  params.batch_size;
	uint32_t num_samples =  params.num_samples;
	uint32_t dimension   =  params.ndim;

    hazy::vector::FVector<fp_type> &x       = model->weights;
    hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];




	        //threadArgs.compute_times_[tid].ptr->Start();//wzk: pure computation time...

    // calculate which chunk of examples we work on
    size_t start = (num_samples / total) * tid; //GetStartIndex(sampsvec.size, tid, total); 
    size_t end   = (total == tid+1)? num_samples : (start + num_samples/total);//GetEndIndex(sampsvec.size, tid, total);


	//1,  Load the local model from the global model 
    //hazy::vector::CopyInto(x, g_local);
	  hazy::vector::CopyInto_stream(x, g_local);

    if (num_bits <= 8)
    {
	  float b_base           = 256.0;  //65536.0; //1.0; // //2^16 or  1
	  model->batch_step_size = params.step_size/(b_base*b_base); // model->batch_step_size = params.step_size/((float)batch_size*b_base*b_base); 
	  float scale = -model->batch_step_size; ///(float)params.batch_size;

	   unsigned char dest[512+dimension];
	   hazy::vector::FVector<unsigned char> dest_char_vector (dest, dimension);

      for (unsigned i = start; i < end; i++) 
      {
		//2.1: Regroup the data from bitweaving...
		samps[i].Unpack_from_bitweaving(dest_char_vector, num_bits); //hazy::vector::Convert_from_bitweaving(dest_char_vector, samps[i].vector, num_bits);

		//2.2: Compute the loss value...
        fp_type delta;
        delta = scale * (Dot( g_local, dest_char_vector) - samps[i].rating); //0.01;//

        //2.3: Update the local model.
        hazy::vector::ScaleAndAdd(
          g_local,
          dest_char_vector, //sample_char.vector, //sample.vector,
          delta
          );        
      }  
    }
    else if (num_bits <= 16)  
    {
		float b_base			 = 65536.0;  //; //1.0; // //2^16 or  1
		model->batch_step_size = params.step_size/(b_base*b_base); // model->batch_step_size = params.step_size/((float)batch_size*b_base*b_base); 
		float scale = -model->batch_step_size; ///(float)params.batch_size;

		unsigned short dest[512+dimension];
		hazy::vector::FVector<unsigned short>dest_short_vector(dest, dimension);
		
		for (unsigned i = start; i < end; i++) 
		{
			//2.1: Regroup the data from bitweaving...
			samps[i].Unpack_from_bitweaving(dest_short_vector, num_bits); //hazy::vector::Convert_from_bitweaving(dest_char_vector, samps[i].vector, num_bits);
			  
			//2.2: Compute the loss value...
			fp_type delta;
			delta = scale * (Dot( g_local, dest_short_vector) - samps[i].rating);
			  
			//2.3: Update the local model.
			hazy::vector::ScaleAndAdd(
				g_local,
				dest_short_vector, //sample_char.vector, //sample.vector,
				delta
				);		  
		}  
	}
	else if (num_bits <= 32)
    {
      printf("Bits: %d. Not Supported yet...", num_bits);
    }

	//threadArgs.compute_times_[tid].ptr->Pause();//wzk: pure computation time...

}


template< class Model, class Params, class Sample > 
void InitPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
{
	Params const &params = *threadArgs.params_;
	Model *model         =  threadArgs.model_;

	//printf("tid = %d, threads = %d\n", tid, total);
    
    // init in a way that each thread will allocate its own memory
	model->initLocalVars(params.ndim, tid);

	(threadArgs.losses_)[tid]  = 0.0; 
    //threadArgs.losses_[tid].ptr = new double;
    //threadArgs.compute_times_[tid].ptr = new hazy::util::Clock;
    //threadArgs.communicate_times_[tid].ptr = new hazy::util::Clock;
}


int main(int argc, char** argv)
{
  #if defined(_HOGWILD) 
    Hogwild<LinearModel, LinearModelParams, LinearModelSampleBitweaving, LinearModelLoader, Exec> executor;
  	printf("Use Hogwild.......\n");
  #endif

  #ifdef AVX2_EN
    printf("Using AVX!.......\n");  
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

	//Add two parameters here. Number of bits for each region, and huge table enable..
	BitWeavingBase bw_master(szTrainFile, dimension, bits_per_mr, num_samples, huge_page_en ); //false
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


	/////////////////Initialization of Thread pool///////////////////////
    hazy::thread::ThreadPool* threadPool_;
	threadPool_ = new hazy::thread::ThreadPool(nthreads);
	threadPool_->Init();


	ThreadArgs<LinearModel, LinearModelParams, LinearModelSampleBitweaving> args;
	args.model_         = model_; //this->model_;
	args.params_        = &p; //this->params_;
	args.samples_       = p_samp;

	args.losses_        = (double *)malloc(nthreads*sizeof(double)); //
	//args.losses_        = this->losses_;
	//args.compute_times_ = this->compute_times_ ;
 	
 	
	/////////////////Initialization of local gradients///////////////////////
	threadPool_->Execute(args, InitPerThread<LinearModel, LinearModelParams, LinearModelSampleBitweaving>);
	threadPool_->Wait();

	
	double sumCommunicateTime = 0.0;

	hazy::util::Clock compute_clock;

	printf("step_size = %.9f\n", step_size); fflush(stdout);
		  
	for(int e = 0; e <= nepochs; ++e)
	{ 	 
		double avgComputeTime = 0.0;
		if(e > 0)
		{
			if (e == target_epoch) //this->params_->
			{
				PCM_initPerformanceMonitor(&inst_Monitor_Event, NULL);
				PCM_start();
			}

      //printf("begin the %d-th epoch, ", e); fflush(stdout);
			compute_clock.Start();

			//1, Start the training task on the computing threads.
			threadPool_->Execute(args, ModelSyncPerThread<LinearModel, LinearModelParams, LinearModelSampleBitweaving>);
			threadPool_->Wait();
			//this->RunEpoch(trainScan_);

			//2, After the computing thread finishes the computation of local models, the main thread will aggregate the local models from the local threads. 
			//hazy::vector::avg_list(model_->weights, model_->local_gradients, nthreads);
			hazy::vector::avg_list_stream(model_->weights, model_->local_gradients, nthreads);

			avgComputeTime = compute_clock.Stop();
      //printf("end the %d-th epoch,  ", e); fflush(stdout);

			sumCommunicateTime += avgComputeTime;

			if (e == target_epoch) //this->params_->
			{
				PCM_stop();
				printf("=====print the profiling result==========\n");
				PCM_printResults();   
				PCM_cleanup();
			} 


    /////3, Compute the loss for the existing model//////////////////////////////////////////////
    threadPool_->Execute(args, ComputeLossPerThread<LinearModel, LinearModelParams, LinearModelSampleBitweaving>);
    threadPool_->Wait();

		}

        // Sum uf losses from each thread....
    double loss = 0.0;
    for(size_t i = 0; i < nthreads; ++i)
    {
      loss += (args.losses_)[i];
    }

    loss /= num_samples;

    printf( "Epoch: %d, loss: %0.7f, computing_time: %0.7f, sum_time: %0.7f\n", e, (float)loss, avgComputeTime, sumCommunicateTime );
		//ComputeLoss(p_samp, num_samples, target_label, class_model, num_bits);
	}
	//printf("%f\nFinished!\n", totalTime / nepochs);


}
