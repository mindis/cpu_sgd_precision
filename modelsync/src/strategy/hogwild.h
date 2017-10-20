#ifndef _HOGWILD_H
#define _HOGWILD_H

#include "global_macros.h"

#ifndef _NUMA_INIT
#define _NUMA_INIT
#define CPUS_PER_NODE 10 // 10: Withouth HT / 20: With HT
#define NUMA_NODES 4
#endif

#include <unistd.h> 

#include "hazy/scan/binfscan.h"
#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/memscan.h"
#include "hazy/scan/sampleblock.h"


#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#include "hazy/thread/thread_pool-inl.h"
#include "hazy/util/clock.h"
#include "hazy/util/simple_random-inl.h"

#include "utils.h"

#include "types/thread_args.h"
#include "types/aligned_pointer.h"
#include "types/timers_info.h"

#include <stddef.h> 

namespace __executor
{
  template< class Model, class Params, class Sample, class Exec > void ComputeLossPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
  {
    Model *model = threadArgs.model_;
    //Params const &params = *threadArgs.params_;
    size_t *current_batch = threadArgs.current_batch_;
    size_t numElems = threadArgs.actual_num_elements_in_batch;

    hazy::vector::FVector<Sample> const & sampsvec = threadArgs.block_->ex;

    // calculate which chunk of examples we work on
    size_t start = GetStartIndex(numElems, tid, total); 
    size_t end = GetEndIndex(numElems, tid, total);

    if((end - start) == 0)
      return; // DO NOTHING ON THIS THREAD

    // keep const correctness
    Sample const * const samps = sampsvec.values;
    fp_type loss = 0.0;
    // compute the loss for each example
    for (unsigned i = start; i < end; i++)
    {
      /* use this commented function to calculate statistics of x_hat, also uncomment params */
      //fp_type l = Exec::ComputeMetaLoss(samps[i], params);

      fp_type l = Exec::SingleLoss(samps[*current_batch + i], model); // Ignore permutation
      loss += l;
    }
    *threadArgs.losses_[tid].ptr = loss;
  }

  template< class Model, class Params, class Sample, class Exec > void RunHogwildPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
  {
    Model *model                                   = threadArgs.model_;
    Params const &params                           = *threadArgs.params_;
    hazy::vector::FVector<Sample> const & sampsvec = threadArgs.block_->ex;
    Sample * samps                                 = sampsvec.values;
    size_t *perm                                   = threadArgs.block_->perm.values;

  threadArgs.compute_times_[tid].ptr->Start();//wzk: pure computation time...

    // calculate which chunk of examples we work on
    size_t start                        = GetStartIndex(sampsvec.size, tid, total); 
    size_t end                          = GetEndIndex(sampsvec.size, tid, total);
    size_t batch_size                   = params.batch_size;
    size_t * current_batch              = new size_t[batch_size];
    size_t actual_num_elements_in_batch = 0;
	   float b_base                       = samps[0].b_binary_to_value();  //65536.0; //1.0; // //2^16 or  1
	//samps
    model->batch_step_size              = params.step_size/((float)batch_size*b_base*b_base); 

    //for (unsigned i = start; i < end; i++) {
    //  size_t indirect = perm[i];
   //   current_batch[i % batch_size] = indirect;

   //   if((i - start) % batch_size == batch_size - 1 || i == end - 1)
    DECREASING_STEPSIZES_ONLY(model->batch_step_size *= pow(model->k, -params.beta));

  //1,  Load the local model from the global model
    hazy::vector::CopyInto(model->weights, model->local_gradients[tid]);

		 unsigned *p_flag_counter     = &(model->get_flag_counter(tid));
     size_t num_threads           = model->nthreads_; //always less than 32...
     size_t num_threads_2_pow     = model->num_threads_2_pow_;
     actual_num_elements_in_batch = end - start; //((i - start) % batch_size) + 1;
        float  scale_factor       = 1.0/(float)num_threads;

    //2, compute the local model from its own samples. 
    Exec::CalcModelUpdate( samps,  &perm[start], actual_num_elements_in_batch, model, params, tid); 
	  //Exec::CalcModelUpdate( samps, current_batch, actual_num_elements_in_batch, model, params, tid);
#if 0
		//3, Accumulate the local model (local gradient) in tree to the global model. 
    if (num_threads > 1)
    {
		  //a, set the finished status for each thread. 
	     model->get_flag_counter(tid) = 1;

      //b, each thread do the aggregation work... 
      //b1, reduce to the last two...
      unsigned check_counter = 1; 
	    for ( size_t tree_base = (num_threads_2_pow>>1); tree_base > 1; tree_base = (tree_base>>1) )
	    {
	    	if (tid < tree_base) //check whether this thread needs to the aggregation. 
        {
          if ((tid + tree_base) >= num_threads)   //the thread in the pair does not exists. 
            model->get_flag_counter(tid) = model->get_flag_counter(tid) + 1; 
          else 
          {
            //poll the thread (tid + tree_base) until its local model is ready.
            while (model->get_flag_counter(tid+tree_base) != check_counter) 
              usleep(100); //sleep for 100us.
            //add the local model (tid+tree_base) to this local model (id) 
            //hazy::vector::CopyInto(model->local_gradients[tid], model->local_gradients[tid+tree_base]); //
            hazy::vector::add(model->local_gradients[tid], model->local_gradients[tid+tree_base]); //do some thing.
            //add 1 to its own counter. 
            model->get_flag_counter(tid) = model->get_flag_counter(tid) + 1; 
          }
        }
        check_counter++;
	    }
      //b2, reduce to the last two...model->local_gradients[0, 1]...
      if (tid == 0)
      {
        if (model->get_flag_counter(1) >= 1)
          hazy::vector::add_mult(model->weights, model->local_gradients[0], model->local_gradients[1], scale_factor); //

        for(unsigned i = 0; i < num_threads; i++)
          model->get_flag_counter(i) = 0;
      }

    }
    else
    {
      hazy::vector::CopyInto(model->local_gradients[tid], model->weights);
    }
#endif
    DECREASING_STEPSIZES_ONLY(model->k += actual_num_elements_in_batch);

    threadArgs.compute_times_[tid].ptr->Pause();//wzk: pure computation time...
  }






  template< class Model, class Params, class Sample, class Exec > void InitPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
  {
    Params const &params = *threadArgs.params_;
    Model *model = threadArgs.model_;

    // init in a way that each thread will allocate its own memory
    model->initLocalVars(params.ndim, tid);
    threadArgs.losses_[tid].ptr = new double;
    threadArgs.compute_times_[tid].ptr = new hazy::util::Clock;
    threadArgs.communicate_times_[tid].ptr = new hazy::util::Clock;
  }


}






template< class Model, class Params, class Sample, class Loader, class Exec>
class Hogwild
{
  public:
    Hogwild() : model_(NULL), params_(NULL), rank_(0)
    {
      wall_clock_.Start();
    }

    virtual ~Hogwild()
    {
      for(size_t i = 0; i < params_->nthreads; ++i)
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
      delete testScan_;
    }

    double NormTwoXMinusXHat()
    {
      hazy::vector::FVector<fp_type> const &x = this->model_->weights;

      fp_type*zeros = new fp_type[this->params_->ndim];
      hazy::vector::FVector<fp_type> *diff = new hazy::vector::FVector<fp_type>(zeros, this->params_->ndim);

      // set to zero
      hazy::vector::Zero(*diff);

      hazy::vector::ScaleAndAdd((*diff), x, 1.0);
      hazy::vector::ScaleAndAdd((*diff), *this->params_->x_hat, -1.0);

      double ret = hazy::vector::Norm2WithoutSquare((*diff));

      delete[] zeros;
      delete[] diff;
      return ret;
    }

    template< class Scan > fp_type ComputeLoss(Scan *scan)
    {
      fp_type loss = 0.0;
      int numSamples = 0;
      scan->Reset();

      test_time_.Start();
      while(scan->HasNext())
      {
        hazy::scan::SampleBlock<Sample> &block = scan->Next();

        // Set losses to 0
        for(size_t i = 0; i < this->params_->nthreads; ++i)
        {
          *losses_[i].ptr = 0;
        }

        ThreadArgs<Model, Params, Sample> args;
        args.model_ = this->model_;
        args.params_ = this->params_;
        args.rank_ = this->rank_;
        args.losses_ = this->losses_;
        args.block_ = &block;

        size_t current_batch[1];
        current_batch[0] = 0;

        args.actual_num_elements_in_batch = block.ex.size;
        args.current_batch_ = current_batch;
        numSamples += block.ex.size;

        // Run on thread pool
        threadPool_->Execute(args, __executor::ComputeLossPerThread<Model, Params, Sample, Exec>);
        threadPool_->Wait();

        // Sum uf losses
        for(size_t i = 0; i < this->params_->nthreads; ++i)
        {
          loss += *losses_[i].ptr;
        }
      }
      test_time_.Stop();

      if(numSamples == 0)
        return 0.0;

      return loss / numSamples;
    }

	//denormalization to binary classification...
	template< class Sample > void b_normalize(hazy::vector::FVector<Sample> &samps)
	{
		unsigned class_model    = this->params_->class_model;	//	
		unsigned target_label   = this->params_->target_label; //intger.
		bool normalize_enable   = class_model&1;     //first bit..
		bool toMinus1_1         = (class_model>>1)&1;//second bit..
		float base              = samps[0].b_binary_to_value();
			
		float not_targeted      = toMinus1_1? (base*(-1.0)): 0.0;
		unsigned num_samps      = samps.size;

		//printf("base = %f, class_model = %d, normalize_enable = %d, toMinus1_1=%d, num_samps= %d\n", base, class_model, normalize_enable, toMinus1_1, num_samps );
		
		if (normalize_enable)
		{
			for (int i = 0; i < num_samps; i++)
			{
				//if(i < 10)
					//printf("%d: original %f, ", i, samps[i].value);
				if (samps[i].value != target_label) //not the target label, assign to -1 or 0. 
					samps[i].value = not_targeted;
				else
					samps[i].value = base;//1.0;             //1.0 for the targeted label...
				//if(i < 10)
					//printf(" now %f", samps[i].value);
			}
		}
		return;
	}
    template< class Scan > void RunEpoch(Scan *scan)
    {
      scan->Reset();

      train_time_.Start();
      epoch_time_.Start();

      while(scan->HasNext())
      {
        hazy::scan::SampleBlock<Sample> &block = scan->Next();

        ThreadArgs<Model, Params, Sample> args;
        args.model_ = this->model_;
        args.params_ = this->params_;
        args.rank_ = this->rank_;
        args.block_ = &block;
        args.compute_times_ = this->compute_times_ ;
        args.communicate_times_ = this->communicate_times_ ;

        threadPool_->Execute(args, __executor::RunHogwildPerThread<Model, Params, Sample, Exec>);
        threadPool_->Wait();
      }

	  unsigned num_threads = this->model_->nthreads_;
	  hazy::vector::avg_list(this->model_->weights, this->model_->local_gradients, num_threads);
/*	   
	  //accumulate the local models to the global model...
	  for (unsigned i = 0; i < this->model_->weights.size; i++)
	  {
	  	(this->model_->weights)[i] = 0.0;
	  	for(unsigned j = 0; j < num_threads; j++)
		  (this->model_->weights)[i] += (this->model_->local_gradients[j])[i];

		(this->model_->weights)[i] = (this->model_->weights)[i] / (float)num_threads;
	  }
*/	  	
      epoch_time_.Stop();
      train_time_.Pause();
    }

    void Init(char *szTrainFile, char *szTestFile, char *szMetadataFile, bool loadBinary, bool matlab_tsv, int dimension, unsigned nthreads)
    {
      hazy::vector::FVector<Sample> train_samps;
      hazy::vector::FVector<Sample> test_samps;
      hazy::vector::FVector<Sample> metadata;

      printDebug(rank_, "Loading training samples from '%s'", szTrainFile);

      if (loadBinary) {
        hazy::scan::BinaryFileScanner scan(szTrainFile);
        Loader::LoadSamples(scan, train_samps, dimension);
      } else if (matlab_tsv) {
        hazy::scan::MatlabTSVFileScanner scan(szTrainFile);
        Loader::LoadSamples(scan, train_samps, dimension);
      } else {
        hazy::scan::TSVFileScanner scan(szTrainFile);
        Loader::LoadSamples(scan, train_samps, dimension);
      }

      printDebug(rank_, "Loading test samples from '%s'", szTestFile);
      if (matlab_tsv) {
        hazy::scan::MatlabTSVFileScanner scantest(szTestFile);
        Loader::LoadSamples(scantest, test_samps, dimension);
      } else {
        hazy::scan::TSVFileScanner scantest(szTestFile);
        Loader::LoadSamples(scantest, test_samps, dimension);
      }

      //hazy::scan::TSVFileScanner scan_metadata(szMetadataFile);
      //Loader::LoadSamples(scan_metadata, metadata, dimension);

      printDebug(rank_, "Loaded %lu training samples", train_samps.size);
      printDebug(rank_, "Loaded %lu test samples", test_samps.size);

      b_normalize(train_samps);
      b_normalize(test_samps);
      //printDebug(rank_, "After b normalization. \n");

      params_->ndim = dimension;
      params_->numSamplesProc = train_samps.size;
      params_->nthreads = nthreads;

      fp_type *d = new fp_type[dimension];
      params_->x_hat= new hazy::vector::FVector<fp_type>(d, dimension);
      hazy::vector::Zero(*params_->x_hat);
      //hazy::vector::ScaleAndAdd(*params_->x_hat, metadata.values[0].vector, 1.0);

      model_ = new Model(dimension, nthreads);

      //trainScan_ = new hazy::scan::MemoryScan< Sample >(train_samps);
      //testScan_ = new hazy::scan::MemoryScan< Sample >(test_samps);
      //trainScan_ = new hazy::scan::MemoryScanNoPermutation< Sample >(train_samps);
      //testScan_ = new hazy::scan::MemoryScanNoPermutation< Sample >(test_samps);
      trainScan_  = new hazy::scan::MemoryScanPermuteValues< Sample >(train_samps);
      testScan_   = new hazy::scan::MemoryScanPermuteValues< Sample >(test_samps);

      // Get max number of samples on any worker
      params_->maxSamplesProc = train_samps.size;
      params_->totalNumSamples = train_samps.size;

      if(this->params_->batch_size < 1)
      {
        // Vanilla gradient descent
        this->params_->batch_size = this->params_->maxSamplesProc;
      }

      DECREASING_STEPSIZES_ONLY(model_->k = 1);
      //DECREASING_STEPSIZES_ONLY(model_->k = params_->batch_size > 1 ? (params_->batch_size / 2) : 1);

      // Init threadpool

      //Cache aligned allocation
      losses_ = (AlignedPointer<double>*)aligned_alloc(CACHE_LINE_SIZE, nthreads * sizeof(AlignedPointer<double>));
      compute_times_ = (AlignedPointer<hazy::util::Clock>*)aligned_alloc(CACHE_LINE_SIZE, nthreads * sizeof(AlignedPointer<hazy::util::Clock>));
      communicate_times_ = (AlignedPointer<hazy::util::Clock>*)aligned_alloc(CACHE_LINE_SIZE, nthreads * sizeof(AlignedPointer<hazy::util::Clock>));
      printDebug(rank_, "Run threadpool with %u threads", nthreads);

      threadPool_ = new hazy::thread::ThreadPool(nthreads);
      threadPool_->Init();
	  
      //printDebug(rank_, "After threadPool_init\n");

      // Init 
      ThreadArgs<Model, Params, Sample> args;
      args.model_ = this->model_;
      args.params_ = this->params_;
      args.losses_ = this->losses_;
      args.compute_times_ = this->compute_times_ ;
      args.communicate_times_ = this->communicate_times_ ;

      threadPool_->Execute(args, __executor::InitPerThread<Model, Params, Sample, Exec>);
     // printDebug(rank_, "After threadPool_Execute\n");

      threadPool_->Wait();
     // printDebug(rank_, "After threadPool_WAIT\n");	  
    }

    void Run(int nepochs)
    {
      //printDebug(rank_, "beginning In run, nepochs = %d\n", nepochs);	 
	  
      double totalTime = 0.0;
	  double pure_total_computing_time = 0.0;
	  printf("this->params_->step_size = %.9f\n", this->params_->step_size);
	  
      for(int e = 0; e <= nepochs; ++e)
      {      
        double avgComputeTime = 0.0;
        double avgCommunicateTime = 0.0;
        if(e > 0)
        {
          this->RunEpoch(trainScan_);

          // Calc average and reset compute and communicate timers
          for(unsigned i = 0; i < params_->nthreads; ++i)
          {
            avgComputeTime += compute_times_[i].ptr->value;
            avgCommunicateTime += communicate_times_[i].ptr->value;
            compute_times_[i].ptr->Reset();     // Force reset
            communicate_times_[i].ptr->Reset(); // Force reset
          }
          if(params_->nthreads > 0)
          {
            avgComputeTime /= params_->nthreads;
            avgCommunicateTime /= params_->nthreads;
          }
		  pure_total_computing_time += avgComputeTime;
        }
		
#ifdef _EXPBACKOFF_STEPSIZES
        this->params_->step_size *= this->params_->step_decay;
#endif

        double train_loss = this->ComputeLoss(trainScan_);
        double test_loss = this->ComputeLoss(testScan_);

        totalTime += epoch_time_.value;
/*
        double norm_x_minus_x_hat = this->NormTwoXMinusXHat();
        printf("epoch: %.2d wall_clock: %.7f train_time: %.7f test_time: %.7f epoch_time: %.7f compute_time: %.7f communicate_time: %.7f train_loss: %.7f test_loss: %.7f norm_x_minus_x_hat: %.7f\n", 
            e,
            wall_clock_.Read(),
            train_time_.value,
            test_time_.value,
            epoch_time_.value,
            avgComputeTime,
            avgCommunicateTime,
            train_loss,
            test_loss,
            norm_x_minus_x_hat
            );
*/
        printf("epoch_%.2d::::::::: train_time (total, each): with_thread_sync(%.7f, %.7f), without_thread_sync(%.7f, %.7f)  train_loss: %.7f test_loss: %.7f\n", //communicate_time: %.7f
            e,
            train_time_.value,
            epoch_time_.value,
            pure_total_computing_time,
            avgComputeTime,
            //avgCommunicateTime,
            train_loss,
            test_loss
            );
        fflush(stdout);
      }

      printf("%f\nFinished!\n", totalTime / nepochs);
    }

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
};

#endif
