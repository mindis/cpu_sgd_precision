#ifndef _EXECUTOR_H
#define _EXECUTOR_H

#define ROOT_ID 0 // For printing the results (not parameter server stuff)
#include "global_macros.h"

#include "mpi.h"

#include "scan/mpifscan.h"
#include "hazy/scan/binfscan.h"
#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/memscan.h"
#include "hazy/scan/sampleblock.h"

#include "hazy/vector/operations-inl.h"
#include "hazy/vector/scale_add-inl.h"

#include "hazy/thread/thread_pool-inl.h"
#include "hazy/util/clock.h"

#ifndef LOAD_FILE_PER_WORKER
#include "hazy/util/simple_random-inl.h"
#endif

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

  template< class Model, class Params, class Sample, class Exec > void ComputeModelUpdatesPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
  {
    Model *model = threadArgs.model_;
    Params const &params = *threadArgs.params_;
    size_t *current_batch = threadArgs.current_batch_;
    size_t numElems = threadArgs.actual_num_elements_in_batch;

    hazy::vector::FVector<Sample> const & sampsvec = threadArgs.block_->ex;
    hazy::vector::Zero(model->local_gradients[tid]);

    // calculate which chunk of examples we work on
    size_t start = GetStartIndex(numElems, tid, total); 
    size_t end = GetEndIndex(numElems, tid, total);

    if((end - start) == 0)
    {
      return; // DO NOTHING ON THIS THREAD
    }

    // keep const correctness
    Sample const * const samps = sampsvec.values;
    // compute the gradient for the samples
    Exec::CalcModelUpdate( samps, current_batch + start, end - start, model, params, tid);
  }

  template< class Model, class Params, class Sample, class Exec > void InitPerThread(ThreadArgs<Model, Params, Sample> &threadArgs, unsigned tid, unsigned total)
  {
    Params const &params = *threadArgs.params_;
    Model *model = threadArgs.model_;

    // init in a way that each thread will allocate its own memory
    model->initLocalVars(params.ndim, tid);
    threadArgs.losses_[tid].ptr = new double;
  }
}

namespace __mpi_user_func
{
  void timersInfoReduce ( TimersInfo *in, TimersInfo *out, int *len, MPI_Datatype *dataType)
  {
    for(int i = 0; i < *len; ++i)
    {
      out[i].epochTime = std::max(in[i].epochTime, out[i].epochTime);
      //out[i].epochTime = in[i].epochTime + out[i].epochTime;
      out[i].trainTime = std::max(in[i].trainTime, out[i].trainTime);
      //out[i].computeTime = std::max(in[i].computeTime, out[i].computeTime);
      //out[i].communicateTime = std::max(in[i].communicateTime, out[i].communicateTime);
      out[i].computeTime = in[i].computeTime + out[i].computeTime;
      out[i].communicateTime = in[i].communicateTime + out[i].communicateTime;
      //out[i].testTime = std::max(in[i].testTime, out[i].testTime);
      out[i].testTime = in[i].testTime + out[i].testTime;
    }
  }
}

template< class Model, class Params, class Sample, class Loader, class Exec>
class Executor
{
  public:
    Executor() : model_(NULL), params_(NULL),runThreadPool_(true), rank_(0), world_size_(0), workerNumber_(0), numberOfWorkers_(0) 
  {
    //MPI_Init(NULL, NULL);
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

    if (provided != MPI_THREAD_FUNNELED)
    {
      printf("MPI Thread Funneled not supported!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    wall_clock_.Start();

    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    // Initialize custom MPI_Datatype

    const int nitems = 5;
    int blocklengths[nitems] = {1, 1, 1, 1, 1};
    MPI_Datatype types[nitems] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(TimersInfo, epochTime);
    offsets[1] = offsetof(TimersInfo, trainTime);
    offsets[2] = offsetof(TimersInfo, computeTime);
    offsets[3] = offsetof(TimersInfo, communicateTime);
    offsets[4] = offsetof(TimersInfo, testTime);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_timers_info_type);
    MPI_Type_commit(&mpi_timers_info_type);

    // Initialize custom MPI_Op
    MPI_Op_create((MPI_User_function *) __mpi_user_func::timersInfoReduce, 1, &mpi_reduce_timers_info_op);
  }

    virtual ~Executor()
    {
      if(runThreadPool_)
      {
        for(size_t i = 0; i < params_->nthreads; ++i)
        {
          delete losses_[i].ptr;
        }
        free(losses_);

        threadPool_->Join();
        delete threadPool_;
      }
      delete model_;
      delete trainScan_;
      delete testScan_;
      MPI_Op_free(&mpi_reduce_timers_info_op);
      MPI_Type_free(&mpi_timers_info_type);
      MPI_Finalize();
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

        if(runThreadPool_)
        {
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
        else
        {
          for(unsigned i = 0; i < block.ex.size; ++i)
          {
            fp_type l = Exec::SingleLoss(block.ex.values[i], this->model_);
            loss += l;
          }
          numSamples += block.ex.size;
        }
      }
      test_time_.Stop();

      int reducedNumSamples = 0;
      MPI_Reduce(&numSamples, &reducedNumSamples, 1, MPI_INT, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
      fp_type reducedLoss = 0.0;
      MPI_Reduce(&loss, &reducedLoss, 1, MPI_FP_TYPE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);

      if(reducedNumSamples == 0)
        return 0.0;

      // Calculate lasso regularization loss
      reducedLoss /= reducedNumSamples;
      double reg = 0.0;
      for(uint64_t i = 0; i < this->model_->weights.size; ++i)
      {
        if(this->model_->weights.values[i] > 0)
        {
          reg += this->model_->weights.values[i];
        }
        else if(this->model_->weights.values[i] < 0)
        {
          reg -= this->model_->weights.values[i];
        }
      }

      return reducedLoss + (this->params_->lasso_regularizer * reg);
    }

    template< class Scan > void RunEpoch(int epoch_nr, Scan *scan)
    {
      MPI_Barrier(MPI_COMM_WORLD);

      if(this->numberOfWorkers_ > 1)
      {
        this->PreEpoch(this->communicate_time_);
      }

      scan->Reset();

      train_time_.Start();
      epoch_time_.Start();

      size_t * current_batch = new size_t[params_->batch_size];
      size_t actual_num_elements_in_batch = 0;

      while(scan->HasNext())
      {
        hazy::scan::SampleBlock<Sample> &block = scan->Next();

        ThreadArgs<Model, Params, Sample> args;
        args.model_ = this->model_;
        args.params_ = this->params_;
        args.rank_ = this->rank_;
        args.block_ = &block;

        // TODO Better way to be able to have the same number of GetModel / CalcModelUpdate and SendModelUpdate for each Process regardless of the number of training samples (Should only be necessary in the synchronized case)
        uint64_t numElements = block.ex.size;
        uint64_t maxNum = params_->maxSamplesProc;

        unsigned iterations = (maxNum / params_->batch_size);
        if(maxNum % params_->batch_size != 0)
          iterations++;

        for (unsigned b = 0; b < iterations; ++b)
        {

          this->model_->batch_step_size = this->params_->step_size; 
          DECREASING_STEPSIZES_ONLY(this->model_->batch_step_size *= pow(model_->k, -params_->beta));
          //DECREASING_STEPSIZES_ONLY(printDebug(rank_, "Updated the batch step size to %f (with k: %lu)!", this->model_->batch_step_size, model_->k));

          actual_num_elements_in_batch = 0;
          unsigned batchIndex = b*params_->batch_size;

          for (unsigned i = 0; i < params_->batch_size; ++i)
          {
            unsigned elem = batchIndex + i;
            if(elem < numElements)
            {
              size_t indirect = block.perm[elem];
              current_batch[i] = indirect;
              actual_num_elements_in_batch++;
            }
          }

          if(this->numberOfWorkers_ > 1)
          {
            this->GetModel(this->communicate_time_);
          }

          if(runThreadPool_)
          {
            args.actual_num_elements_in_batch = actual_num_elements_in_batch;
            args.current_batch_ = current_batch;

            compute_time_.Start();
            // Run on thread pool
            threadPool_->Execute(args, __executor::ComputeModelUpdatesPerThread<Model, Params, Sample, Exec>);
            threadPool_->Wait();

            // Sum up gradients
            for(size_t i = 0; i < params_->nthreads; ++i)
            {
              hazy::vector::ScaleAndAdd(model_->gradient, model_->local_gradients[i], 1.0);
            }

            compute_time_.Pause();
          }
          else
          {
            compute_time_.Start();
            Exec::CalcModelUpdate( block.ex.values, current_batch, actual_num_elements_in_batch, model_, *params_, 0);
            compute_time_.Pause();
          }

          // Print gradient
#ifdef _PRINT_GRADIENT
          int nr = ((epoch_nr-1) * iterations) + b;
          for(size_t i = 0; i < this->model_->gradient.size; ++i)
          {
            if(std::abs(this->model_->gradient.values[i]) >= epsilon) {
              std::cout << nr << "\t" << i << "\t" << this->model_->gradient.values[i] << std::endl;
            }
          }
#endif

          // Add lasso regularizer to gradient //TODO omit division by totalNumSamples for logistic regression
          float scale = -(model_->batch_step_size/params_->totalNumSamples) * params_->lasso_regularizer;

          for(uint64_t i = 0; i < this->model_->gradient.size; ++i)
          {
            if(this->model_->weights.values[i] > 0)
            {
              this->model_->gradient[i] += scale;
            }
            else if(this->model_->weights.values[i] < 0)
            {
              this->model_->gradient[i] -= scale;
            }
          }

          if(this->numberOfWorkers_ > 1)
          {
            this->SendModelUpdate(this->communicate_time_);

            // Reset gradient
            hazy::vector::Zero(this->model_->gradient);
          }
          else
          {
            hazy::vector::ScaleAndAdd(this->model_->weights, this->model_->gradient, 1.0);
            // Reset gradient
            hazy::vector::Zero(this->model_->gradient);
          }

          DECREASING_STEPSIZES_ONLY(model_->k += actual_num_elements_in_batch);
          //DECREASING_STEPSIZES_ONLY(model_->k += params_->batch_size);
        }
      }

      delete[] current_batch;

      if(this->numberOfWorkers_ > 1)
      {
        // Get model in order to have consistent models for loss computation
        this->GetModel(this->communicate_time_);

        // Post epoch for Async - Parameter server implementation
        this->PostEpoch(communicate_time_);
      }

      epoch_time_.Stop();
      train_time_.Pause();
    }

    void Init(char *szTrainFile, char *szTestFile, char *szMetadataFile, bool loadBinary, bool matlab_tsv, int dimension, unsigned nthreads)
    {
      hazy::vector::FVector<Sample> train_samps;
      hazy::vector::FVector<Sample> test_samps;
      hazy::vector::FVector<Sample> metadata;

      this->workerNumber_ = GetWorkerNumber();
      int worker = this->workerNumber_ != -1 ? 1 : 0;

      // Get total number of workers
      MPI_Allreduce(&worker, &numberOfWorkers_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(numberOfWorkers_ < 1)
      {
        printf("At least one worker has to be specified!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      if(rank_ == 0)
      {
        printDebug(rank_, "Splitting the work amongst %i workers", numberOfWorkers_);
      }

#ifdef LOAD_FILE_PER_WORKER
      if(this->workerNumber_ != -1) {
        char** trainFileName = new char*;
        char** testFileName = new char*;

        if(this->numberOfWorkers_ < 2) {
          copyString(szTrainFile, trainFileName);
          copyString(szTestFile, testFileName);
        } else {
          getSplitNameOfFile(szTrainFile, this->workerNumber_, this->numberOfWorkers_, trainFileName);
          getSplitNameOfFile(szTestFile, this->workerNumber_, this->numberOfWorkers_, testFileName);
        }

        printDebug(rank_, "Loading training samples from '%s'", *trainFileName);
        if(!fileExists(*trainFileName)) {
          printf("File '%s' does not exist!", *trainFileName);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (loadBinary) {
          hazy::scan::BinaryFileScanner scan(*trainFileName);
          Loader::LoadSamples(scan, train_samps, dimension);
        } else if (matlab_tsv) {
          hazy::scan::MatlabTSVFileScanner scan(*trainFileName);
          Loader::LoadSamples(scan, train_samps, dimension);
        } else {
          hazy::scan::TSVFileScanner scan(*trainFileName);
          Loader::LoadSamples(scan, train_samps, dimension);
        }

        printDebug(rank_, "Loading test samples from '%s'", *testFileName);
        if(!fileExists(*testFileName)) {
          printf("File '%s' does not exist!", *testFileName);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (loadBinary) {
          hazy::scan::BinaryFileScanner scantest(*testFileName);
          Loader::LoadSamples(scantest, test_samps, dimension);
        } else if (matlab_tsv) {
          hazy::scan::MatlabTSVFileScanner scantest(*testFileName);
          Loader::LoadSamples(scantest, test_samps, dimension);
        } else {
          hazy::scan::TSVFileScanner scantest(*testFileName);
          Loader::LoadSamples(scantest, test_samps, dimension);
        }

        delete[] *trainFileName;
        delete[] *testFileName;

        delete trainFileName;
        delete testFileName;
      }
#else
      if (this->numberOfWorkers_ < 2) {
        if (this->workerNumber_ != -1) {
          printDebug(rank_, "Loading training samples from '%s'", szTrainFile);
          if(!fileExists(szTrainFile)) {
            printf("File '%s' does not exist!", szTrainFile);
            MPI_Abort(MPI_COMM_WORLD, 2);
          }

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
          if(!fileExists(szTestFile)) {
            printf("File '%s' does not exist!", szTestFile);
            MPI_Abort(MPI_COMM_WORLD, 2);
          }

          if (loadBinary) {
            hazy::scan::BinaryFileScanner scantest(szTestFile);
            Loader::LoadSamples(scantest, test_samps, dimension);
          } else if (matlab_tsv) {
            hazy::scan::MatlabTSVFileScanner scantest(szTestFile);
            Loader::LoadSamples(scantest, test_samps, dimension);
          } else {
            hazy::scan::TSVFileScanner scantest(szTestFile);
            Loader::LoadSamples(scantest, test_samps, dimension);
          }
        }
      } else {
        // Use MPI-IO only on binary files
        if (!loadBinary) {
          printf("MPI-IO only supported with binary file format!");
          MPI_Abort(MPI_COMM_WORLD, 3);
        }

        // Get train split file name
        char** trainSeekPointsFileName = new char*;
        char** testSeekPointsFileName = new char*;

        std::stringstream s;
        s << "_seek_points_" << this->numberOfWorkers_;
        changeFilename(szTrainFile, s.str(), "bin", trainSeekPointsFileName);

        printDebug(rank_, "Input seek points train filename: %s", *trainSeekPointsFileName);
        if(!fileExists(*trainSeekPointsFileName)) {
          printf("File '%s' does not exist!", *trainSeekPointsFileName);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        changeFilename(szTestFile, s.str(), "bin", testSeekPointsFileName);

        printDebug(rank_, "Input seek points test filename: %s", *testSeekPointsFileName);
        if(!fileExists(*testSeekPointsFileName)) {
          printf("File '%s' does not exist!", *testSeekPointsFileName);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        // Get seek points
        MPI_File fhTrain;
        MPI_File_open(MPI_COMM_WORLD, *trainSeekPointsFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhTrain);

        uint64_t vals[2];
        if(this->workerNumber_ < 0) {
          vals[0] = 0;
          vals[1] = 0;
        }
        else {
          MPI_File_seek(fhTrain, this->workerNumber_ * sizeof(uint64_t) * 2, MPI_SEEK_SET );
          MPI_File_read(fhTrain, &vals, 2, MPI_UINT64_T, MPI_STATUS_IGNORE);
        }

        MPI_File_close(&fhTrain);

        printDebug(rank_, "Loading training samples (starting: %lu) from '%s'", vals[0], szTrainFile);
        if(!fileExists(szTrainFile)) {
          printf("File '%s' does not exist!", szTrainFile);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        MpiFileScanner scan(szTrainFile, vals[0], vals[1]);
        Loader::LoadSamples(scan, train_samps, dimension);

        // Get seek points
        MPI_File fhTest;
        MPI_File_open(MPI_COMM_WORLD, *testSeekPointsFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhTest);

        if(this->workerNumber_ < 0) {
          vals[0] = 0;
          vals[1] = 0;
        }
        else {
          MPI_File_seek(fhTest, this->workerNumber_ * sizeof(uint64_t) * 2, MPI_SEEK_SET );
          MPI_File_read(fhTest, &vals, 2, MPI_UINT64_T, MPI_STATUS_IGNORE);
        }

        MPI_File_close(&fhTest);

        printDebug(rank_, "Loading test samples (starting %lu) from '%s'", vals[0], szTestFile);
        if(!fileExists(szTestFile)) {
          printf("File '%s' does not exist!", szTestFile);
          MPI_Abort(MPI_COMM_WORLD, 2);
        }

        MpiFileScanner scantest(szTestFile, vals[0], vals[1]);
        Loader::LoadSamples(scantest, test_samps, dimension);

        delete[] *trainSeekPointsFileName;
        delete[] *testSeekPointsFileName;
        delete trainSeekPointsFileName;
        delete testSeekPointsFileName;
      }

#endif

      hazy::scan::TSVFileScanner scan_metadata(szMetadataFile);
      Loader::LoadSamples(scan_metadata, metadata, dimension);

      printDebug(rank_, "Loaded %lu training samples", train_samps.size);
      printDebug(rank_, "Loaded %lu test samples", test_samps.size);

      MPI_Barrier(MPI_COMM_WORLD);

      params_->ndim = dimension;
      params_->numSamplesProc = train_samps.size;
      params_->nthreads = nthreads;

      fp_type *d = new fp_type[dimension];
      params_->x_hat= new hazy::vector::FVector<fp_type>(d, dimension);
      hazy::vector::Zero(*params_->x_hat);
      hazy::vector::ScaleAndAdd(*params_->x_hat, metadata.values[0].vector, 1.0);

      model_ = new Model(dimension, nthreads);

      trainScan_ = new hazy::scan::MemoryScan< Sample >(train_samps);
      testScan_ = new hazy::scan::MemoryScan< Sample >(test_samps);
      //trainScan_ = new hazy::scan::MemoryScanNoPermutation< Sample >(train_samps);
      //testScan_ = new hazy::scan::MemoryScanNoPermutation< Sample >(test_samps);

      // Get max number of samples on any worker
      MPI_Allreduce(&params_->numSamplesProc, &params_->maxSamplesProc, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
      // Get total number of samples
      MPI_Allreduce(&params_->numSamplesProc, &params_->totalNumSamples, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

      this->params_->batch_size *= nthreads;

      if(this->params_->batch_size < 1)
      {
        // Vanilla gradient descent
        this->params_->batch_size = this->params_->maxSamplesProc;
      }

      DECREASING_STEPSIZES_ONLY(model_->k = 1);
      //DECREASING_STEPSIZES_ONLY(model_->k = params_->batch_size > 1 ? (params_->batch_size / 2) : 1);

      // Init threadpool (with no)
      runThreadPool_ = nthreads > 1;
      if(runThreadPool_)
      {
        //Cache aligned allocation
        losses_ = (AlignedPointer<double>*)aligned_alloc(CACHE_LINE_SIZE, nthreads * sizeof(AlignedPointer<double>));
        printDebug(rank_, "Run threadpool with %u threads (thus %i batch_size for process)", nthreads, this->params_->batch_size);

        threadPool_ = new hazy::thread::ThreadPool(nthreads);
        threadPool_->Init();

        // Init 
        ThreadArgs<Model, Params, Sample> args;
        args.model_ = this->model_;
        args.params_ = this->params_;
        args.losses_ = this->losses_;

        threadPool_->Execute(args, __executor::InitPerThread<Model, Params, Sample, Exec>);
        threadPool_->Wait();
      }

      this->InitStrategy();
    }

    void Run(int nepochs)
    {
#ifndef _PRINT_GRADIENT
      double totalTime = 0.0;
#endif
      for(int e = 0; e <= nepochs; ++e)
      {
        TimersInfo ti;

        if(e > 0)
        {
          this->RunEpoch(e, trainScan_);
          ti.epochTime = epoch_time_.value;
          ti.trainTime = train_time_.value;
          ti.computeTime = compute_time_.value;
          compute_time_.Reset(); // Force reset
          ti.communicateTime = communicate_time_.value;
          communicate_time_.Reset(); // Force reset
        }

#ifdef _EXPBACKOFF_STEPSIZES
        this->params_->step_size *= this->params_->step_decay;
#endif

#ifndef _PRINT_GRADIENT
        double train_loss = this->ComputeLoss(trainScan_);
        double test_loss = this->ComputeLoss(testScan_);

        ti.testTime = test_time_.value;

        TimersInfo reducedTi;
        MPI_Reduce(&ti, &reducedTi, 1, mpi_timers_info_type, mpi_reduce_timers_info_op, ROOT_ID, MPI_COMM_WORLD);
        totalTime += reducedTi.epochTime;

        if(this->rank_ == ROOT_ID)
        {
          double norm_x_minus_x_hat = this->NormTwoXMinusXHat();

          reducedTi.testTime /= numberOfWorkers_;
          reducedTi.computeTime /= numberOfWorkers_;
          reducedTi.communicateTime /= numberOfWorkers_;

          printf("epoch: %.2d wall_clock: %.7f train_time: %.7f test_time: %.7f epoch_time: %.7f compute_time: %.7f communicate_time: %.7f train_loss: %.7f test_loss: %.7f norm_x_minus_x_hat: %.7f\n", 
              e,
              wall_clock_.Read(),
              reducedTi.trainTime,
              reducedTi.testTime,
              reducedTi.epochTime,
              reducedTi.computeTime,
              reducedTi.communicateTime,
              train_loss,
              test_loss,
              norm_x_minus_x_hat
              );
          fflush(stdout);
        }
#endif

        // force all prograss to wait
        MPI_Barrier(MPI_COMM_WORLD);
      }

#ifndef _PRINT_GRADIENT
      if(this->rank_ == ROOT_ID)
      {
        printf("%f\nFinished!\n", totalTime / nepochs);

        //// Print final trained model
        //for(size_t i = 0; i < this->model_->weights.size; ++i)
        //{
        //  std::cout << i << "\t" << (std::abs(this->model_->weights.values[i]) < epsilon ? 0 : this->model_->weights.values[i]) << std::endl;
        //}
      }
#endif
    }

    virtual int GetWorkerNumber() = 0;
    virtual void GetModel(hazy::util::Clock &communicate_timer) = 0;
    virtual void SendModelUpdate(hazy::util::Clock &communicate_timer) = 0;
    virtual void PreEpoch(hazy::util::Clock &communicate_timer) = 0;
    virtual void PostEpoch(hazy::util::Clock &communicate_timer) = 0;
    virtual void InitStrategy() = 0;

    Model *model_;
    Params *params_;

    hazy::thread::ThreadPool* threadPool_;
    bool runThreadPool_;
    AlignedPointer<double>* losses_;

    hazy::util::Clock wall_clock_;
    hazy::util::Clock train_time_;
    hazy::util::Clock test_time_;
    hazy::util::Clock epoch_time_;
    hazy::util::Clock compute_time_;
    hazy::util::Clock communicate_time_;

    int rank_;
    int world_size_;
    int workerNumber_;
    int numberOfWorkers_;

    MPI_Datatype mpi_timers_info_type; 
    MPI_Op mpi_reduce_timers_info_op;

    hazy::scan::MemoryScan< Sample > *trainScan_;
    hazy::scan::MemoryScan< Sample > *testScan_;
    //hazy::scan::MemoryScanNoPermutation< Sample > *trainScan_;
    //hazy::scan::MemoryScanNoPermutation< Sample > *testScan_;
};

#endif
