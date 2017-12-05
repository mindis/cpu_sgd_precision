#ifndef _MODELSYNC_H
#define _MODELSYNC_H

#include "global_macros.h"


#include "hazy/scan/binfscan.h"
#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/memscan.h"
#include "hazy/scan/sampleblock.h"

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

#include "hazy/util/clock.h"
#include "hazy/util/simple_random-inl.h"

//#include "utils.h"

#include "types/thread_args.h"
#include "types/aligned_pointer.h"
#include "types/timers_info.h"

#include <stddef.h> 


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









template< class Model, class Params, class Sample> 
void ComputeLossPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
{
  Model *model         =  threadArgs.model_ ;
    Params const &params = *threadArgs.params_;
    Sample     *samps    =  threadArgs.samples_;
    double    *loss_addr =  threadArgs.losses_;

  unsigned num_bits  =  params.num_bits;
  size_t batch_size  =  params.batch_size;
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


  unsigned num_bits  =  params.num_bits;
  size_t batch_size  =  params.batch_size;
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
    float b_base       = 65536.0;  //; //1.0; // //2^16 or  1
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



template< class Model, class Params, class Sample> //, class Loader, class Exec
class ModelSync
{
  public:
    ModelSync() //: model_(NULL), params_(NULL), rank_(0)
    {
      //wall_clock_.Start();
    }

    virtual ~ModelSync()
    {
/*      for(size_t i = 0; i < params_->nthreads; ++i)
      {
        delete losses_[i].ptr;
        delete compute_times_[i].ptr;
        delete communicate_times_[i].ptr;
      }
      free(losses_);
      free(compute_times_);
      free(communicate_times_);

      threadPool_->Join();
      delete threadPool_;
      delete model_;
      delete trainScan_;
      delete testScan_; */

    }


void Run(Model* model_, Params p, Sample* p_samp, int nepochs)
{
  unsigned nthreads     = p.nthreads;
  uint32_t num_samples  = p.num_samples;
  unsigned target_epoch = p.target_epoch;
  float    step_size    = p.step_size;

  ThreadArgs<LinearModel, LinearModelParams, LinearModelSampleBitweaving> args;
  args.model_         = model_; //this->model_;
  args.params_        = &p; //this->params_;
  args.samples_       = p_samp;


  args.losses_        = (double *)malloc(nthreads*sizeof(double)); //

  /////////////////Initialization of Thread pool///////////////////////
  hazy::thread::ThreadPool* threadPool_;
  threadPool_ = new hazy::thread::ThreadPool(nthreads);
  threadPool_->Init();
  
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

    }
    
    
    /////3, Compute the loss for the existing model//////////////////////////////////////////////
    threadPool_->Execute(args, ComputeLossPerThread<LinearModel, LinearModelParams, LinearModelSampleBitweaving>);
    threadPool_->Wait();

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


  free(args.losses_);


}
/*
    Model *model_;
    Params *params_;

    hazy::thread::ThreadPool* threadPool_;
    AlignedPointer<double>* losses_;
    AlignedPointer<hazy::util::Clock>* compute_times_;
    AlignedPointer<hazy::util::Clock>* communicate_times_;

    hazy::util::Clock wall_clock_;
    hazy::util::Clock train_time_;
    hazy::util::Clock test_time_;
    hazy::util::Clock epoch_time_;
    hazy::util::Clock pure_train_time_;	

    int rank_;

    //hazy::scan::MemoryScan< Sample > *trainScan_;
    //hazy::scan::MemoryScan< Sample > *testScan_;
    //hazy::scan::MemoryScanNoPermutation< Sample > *trainScan_;
    //hazy::scan::MemoryScanNoPermutation< Sample > *testScan_;
    hazy::scan::MemoryScanPermuteValues< Sample > *trainScan_;
    hazy::scan::MemoryScanPermuteValues< Sample > *testScan_;    

*/    
};

#endif
