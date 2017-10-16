#ifndef _LINEARMODEL_H
#define _LINEARMODEL_H

#include "hazy/vector/fvector.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/operations-inl.h"

#ifdef AVX2_EN
#include "hazy/vector/scale_add-inl_avx2.h"
#else
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
    weights.values = new fp_type[dim];
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
    delete[] weights.values;
    delete[] gradient.values;
  }
};

struct LinearModelParams
{
  unsigned class_model;       //first bit: enable binary classification, second bit: -1 or 0  Default: 0"
  unsigned target_label;      //Target label to be identified. default:1
  
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
  hazy::vector::FVector<fp_type> *x_hat;

  LinearModelParams(fp_type stepsize, fp_type stepdecay) : step_size(stepsize), step_decay(stepdecay) { }
};


//////////////////////////////////////////char//////////////////////////////////////////////////////////
struct LinearModelSample_char
{
  fp_type value;            //!< rating of this example
  SPARSE_ONLY(hazy::vector::SVector<unsigned char> vector;); //!< feature vector
  DENSE_ONLY( hazy::vector::FVector<unsigned char> vector;); //!< feature vector

  LinearModelSample_char()
  { }

#ifdef _SPARSE
  LinearModelSample_char(
    fp_type val,
    unsigned char  *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val), vector(values, index, len) { }
#else
  LinearModelSample_char(
    fp_type val,
    unsigned char  *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val) {

    hazy::vector::SVector<unsigned char> sparse_vector(values, index, len);

    unsigned char *zeros = new unsigned char[dimension];
    hazy::vector::FVector<unsigned char> *temp_vector = new hazy::vector::FVector<unsigned char>(zeros, dimension);

    // set to zero
    hazy::vector::Zero(*temp_vector);
    
    hazy::vector::ScaleAndAdd(
		      *temp_vector,
		      sparse_vector,
		      1.0
    );

    vector.size = (*temp_vector).size;
    vector.values = (*temp_vector).values;
  }
#endif

  LinearModelSample_char(const LinearModelSample_char &o) {
    value = o.value;
    vector.values = o.vector.values;
    SPARSE_ONLY(vector.index = o.vector.index;);
    vector.size = o.vector.size;
  }

  float b_binary_to_value()
  {
    //printf("In char binary to value\n");
  	return 256.0;
  }
  void exchange_sample(LinearModelSample_char &exchange)
  {
    //exhange the values.
    unsigned char* this_vector = (unsigned char*)(this->vector.values);
    unsigned char* exch_vector = (unsigned char*)exchange.vector.values;
    unsigned this_size   = this->vector.size;
    unsigned exch_size   = exchange.vector.size;
    unsigned char temp[this_size];
    memcpy(         temp,            (void *) this_vector,   this_size * sizeof(char));
    memcpy((void *) this_vector,     (void *) exch_vector,   this_size * sizeof(char));
    memcpy((void *) exch_vector,          temp,              this_size * sizeof(char));

     //exhange the label.
    fp_type label_tmp;
    label_tmp        = this->value;
    this->value      = exchange.value;      
    exchange.value   = label_tmp;    
  }

};





//////////////////////////////////////////short//////////////////////////////////////////////////////////
struct LinearModelSample_short
{
  fp_type value;            //!< rating of this example
  SPARSE_ONLY(hazy::vector::SVector<unsigned short> vector;); //!< feature vector
  DENSE_ONLY( hazy::vector::FVector<unsigned short> vector;); //!< feature vector

  LinearModelSample_short()
  { }

#ifdef _SPARSE
  LinearModelSample_short(
    fp_type val,
    unsigned short  *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val), vector(values, index, len) { }
#else
  LinearModelSample_short(
    fp_type val,
    unsigned short  *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val) {

    hazy::vector::SVector<unsigned short> sparse_vector(values, index, len);

    unsigned short *zeros = new unsigned short[dimension];
    hazy::vector::FVector<unsigned short> *temp_vector = new hazy::vector::FVector<unsigned short>(zeros, dimension);

    // set to zero
    hazy::vector::Zero(*temp_vector);
    
    hazy::vector::ScaleAndAdd(
		      *temp_vector,
		      sparse_vector,
		      1.0
    );

    vector.size = (*temp_vector).size;
    vector.values = (*temp_vector).values;
  }
#endif

  LinearModelSample_short(const LinearModelSample_short &o) {
    value = o.value;
    vector.values = o.vector.values;
    SPARSE_ONLY(vector.index = o.vector.index;);
    vector.size = o.vector.size;
  }

  float b_binary_to_value()
  {
  	return 65536.0;
  }
  void exchange_sample(LinearModelSample_short &exchange)
  {
    //exhange the values.
    unsigned short* this_vector = (unsigned short*)(this->vector.values);
    unsigned short* exch_vector = (unsigned short*)exchange.vector.values;
    unsigned this_size   = this->vector.size;
    unsigned exch_size   = exchange.vector.size;
    short temp[this_size];
    memcpy(         temp,            (void *) this_vector,   this_size * sizeof(short));
    memcpy((void *) this_vector,     (void *) exch_vector,   this_size * sizeof(short));
    memcpy((void *) exch_vector,          temp,              this_size * sizeof(short));

     //exhange the label.
    fp_type label_tmp;
    label_tmp        = this->value;
    this->value      = exchange.value;      
    exchange.value   = label_tmp;    
  }

};










//////////////////////////////////////////float//////////////////////////////////////////////////////////
struct LinearModelSample
{
  fp_type value;            //!< rating of this example
  SPARSE_ONLY(hazy::vector::SVector<const fp_type> vector;); //!< feature vector
  DENSE_ONLY( hazy::vector::FVector<const fp_type> vector;); //!< feature vector

  LinearModelSample()
  { }

#ifdef _SPARSE
  LinearModelSample(
    fp_type val,
    fp_type const *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val), vector(values, index, len) { }
#else
  LinearModelSample(
    fp_type val,
    fp_type const *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val) {

    hazy::vector::SVector<const fp_type> sparse_vector(values, index, len);

    fp_type *zeros = new fp_type[dimension];
    hazy::vector::FVector<fp_type> *temp_vector = new hazy::vector::FVector<fp_type>(zeros, dimension);

    // set to zero
    hazy::vector::Zero(*temp_vector);
    
    hazy::vector::ScaleAndAdd(
		      *temp_vector,
		      sparse_vector,
		      1.0
    );

    vector.size = (*temp_vector).size;
    vector.values = (*temp_vector).values;
  }
#endif

  LinearModelSample(const LinearModelSample &o) {
    value = o.value;
    vector.values = o.vector.values;
    SPARSE_ONLY(vector.index = o.vector.index;);
    vector.size = o.vector.size;
  }

  float b_binary_to_value()
  {
  	return 1.0;
  }

  void exchange_sample(LinearModelSample &exchange)
  {
    //exhange the values.
    fp_type* this_vector = (fp_type*)(this->vector.values);
    fp_type* exch_vector = (fp_type*)exchange.vector.values;
    unsigned this_size   = this->vector.size;
    unsigned exch_size   = exchange.vector.size;
    fp_type temp[this_size];
    memcpy(         temp,            (void *) this_vector,   this_size * sizeof(fp_type));
    memcpy((void *) this_vector,     (void *) exch_vector,   this_size * sizeof(fp_type));
    memcpy((void *) exch_vector,          temp,              this_size * sizeof(fp_type));

     //exhange the label.
    fp_type label_tmp;
    label_tmp        = this->value;
    this->value      = exchange.value;      
    exchange.value   = label_tmp;    
  }

};

#endif
