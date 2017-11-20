#ifndef _LINEARMODEL_H
#define _LINEARMODEL_H

#include "hazy/vector/fvector.h"
#include "hazy/vector/svector.h"

#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif


struct LinearModel
{
  hazy::vector::FVector<fp_type> weights;
  hazy::vector::FVector<fp_type> gradient;
  hazy::vector::FVector<fp_type> * local_gradients;

  float batch_step_size;    //!< current batch step size
  DECREASING_STEPSIZES_ONLY(unsigned long long k;); //!< number of processed element for batch_step_size calculation

  size_t nthreads_;

  explicit LinearModel(unsigned dim, unsigned nthreads)
  {
    nthreads_ = nthreads;
    weights.values = (fp_type*)aligned_alloc(CACHE_LINE_SIZE, dim * sizeof(fp_type)); //new fp_type[dim];
    gradient.values = new fp_type[dim];
    weights.size = dim;
    gradient.size = dim;

    for(unsigned i = 0; i < dim; ++i)
    {
      weights.values[i] = 0;
      gradient.values[i] = 0;
    }

    if(nthreads > 1)
    {
      local_gradients = new hazy::vector::FVector<fp_type>[nthreads];
    }
    else
    {
      local_gradients = &gradient;
    }
  }

  void initLocalVars(unsigned dim, unsigned tid)
  {
    local_gradients[tid].values = (fp_type*)aligned_alloc(CACHE_LINE_SIZE, dim * sizeof(fp_type));
    local_gradients[tid].size = dim;
    for (unsigned i = 0; i < dim; ++i) {
      local_gradients[tid].values[i] = 0;
    }
  }

  ~LinearModel()
  {
    if(nthreads_ > 1)
    {
      for(size_t j = 0; j < nthreads_; j++) {
        free(local_gradients[j].values);
      }
      delete[] local_gradients;
    }
    free(weights.values);//delete[] weights.values;
    delete[] gradient.values;
  }
};

struct LinearModelParams
{
  unsigned class_model;       //first bit: enable binary classification, second bit: -1 or 0  Default: 0"
  unsigned target_label;      //Target label to be identified. default:1
  unsigned target_epoch;      //The index of the epoch where the Intel PCM performs.
  unsigned num_bits;          //Number of bits used for training...
  unsigned model;             //The model...
  uint32_t num_samples;

  unsigned batch_size;        //!< batch size
  float step_size;            //!< stepsize (decayed by step decay at each epoch and/or after each mini batch...)
  float step_decay;           //!< factor to modify step_size by each epoch
  float beta;                 //!< the exponent constant for the stepsizes
  float lasso_regularizer;    //!< lasso regularizer hyperparamter
  uint64_t numSamplesProc;    //!< Number of samples on current worker
  uint64_t totalNumSamples;   //!< Total number of samples
  uint64_t maxSamplesProc;    //!< Max number on any worker
  unsigned ndim;              //!< number of features, length of degrees
  unsigned nthreads;          //!< Number of threads on working processes
  unsigned quantizationLevel; //!< Quantization level (only affects LinReg and if quantization is > 0)
  unsigned quantization;      //!< 0: No quantization / 1: Quatize samples / 2: quantize gradients / 3: quantize samples and gradient / 4: Quantize all Model / Sample and Gradient)
  //hazy::vector::FVector<fp_type> *x_hat;

  LinearModelParams(fp_type stepsize, fp_type stepdecay) : step_size(stepsize), step_decay(stepdecay) { }
};

#endif
