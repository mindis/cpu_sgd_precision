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
  unsigned target_epoch;      //The index of the epoch where the Intel PCM performs.
  unsigned num_bits;          //Number of bits used for training...
  
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



#if 1

#define uint32_t unsigned int
#define BITS_OF_ONE_CACHE_LINE 512

//Suppose the size of each value of training dataset is 32-bit, always true for our case...
uint32_t compute_num_CLs_per_sample(uint32_t dr_numFeatures) {
  //With the chunk of 512 features...
  uint32_t main_num           = (dr_numFeatures/BITS_OF_ONE_CACHE_LINE)*32; //It is for CLs
  uint32_t rem_num            = 0;

  //For the remainder of dr_numFeatures...
  uint32_t remainder_features = dr_numFeatures & (BITS_OF_ONE_CACHE_LINE - 1); 
  if (remainder_features == 0)
    rem_num = 0;
  else if (remainder_features <= 64)
    rem_num = 4;
  else if (remainder_features <= 128) 
    rem_num = 8;
  else if (remainder_features <= 256) 
    rem_num = 16;
  else  
    rem_num = 32;
  //printf("main_num = %d, rem_num = %d\t", main_num, rem_num);
  return main_num + rem_num;
}

void bitweaving_on_each_sample(uint32_t *dest, uint32_t *src, uint32_t numFeatures) 
{
  //Compute the number of CLs for each sample...
  int num_CLs_per_sample     = compute_num_CLs_per_sample(numFeatures);
  //printf("num_CLs_per_sample = %d\n", num_CLs_per_sample);
  //uint32_t *a_fpga_tmp       = a_bitweaving_fpga;
  uint32_t address_index     = 0;
  int num_features_main      = (numFeatures/BITS_OF_ONE_CACHE_LINE)*BITS_OF_ONE_CACHE_LINE;  

    //Deal with the main part of dr_numFeatures.
    for (int j = 0; j < num_features_main; j += BITS_OF_ONE_CACHE_LINE)
    {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      //1: initilization off tmp buffer..
      for (int k = 0; k < BITS_OF_ONE_CACHE_LINE; k++)
      {
        tmp_buffer[k] = src[j + k];
        //printf("src[%d] = 0x%8x\t", j+k, src[j + k]);
      }  


      //2: focus on the data from index: j...
      for (int k = 0; k < 32; k++)
      { 
        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE/32] = {0};  //16 ints == 512 bits...
        //2.1: re-order the data according to the bit-level...
        for (int m = 0; m < BITS_OF_ONE_CACHE_LINE; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }
        //2.2: store the bit-level result back to the memory...
        dest[address_index++] = result_buffer[0]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[1]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[2]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[3]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[4]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[5]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[6]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[7]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[8]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[9]; //printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[10];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[11];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[12];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[13];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[14];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
        dest[address_index++] = result_buffer[15];//printf("dest[%d] = 0x%8x\t", address_index-1, dest[address_index-1]);
      }
    }

    //Deal with the remainder of features, with the index from j...
    uint32_t num_r_f = numFeatures - num_features_main;
    //handle the remainder....It is important...
    if (num_r_f > 0)
    {
      uint32_t tmp_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
      for (int k = 0; k < num_r_f; k++)
        tmp_buffer[k] = src[num_features_main + k]; //j is the existing index...

      for (int k = 0; k < 32; k++) //64 bits for each bit...
      {
        uint32_t result_buffer[BITS_OF_ONE_CACHE_LINE] = {0};
        for (int m = 0; m < num_r_f; m++)
        {
          result_buffer[m>>5] = result_buffer[m>>5] | ((tmp_buffer[m] >>31)<<(m&31));
          tmp_buffer[m]       = tmp_buffer[m] << 1;       
        }
          //1--64 
          dest[address_index++] = result_buffer[0];
          dest[address_index++] = result_buffer[1];

        if (num_r_f > 64)
        { //65--128 
          dest[address_index++] = result_buffer[2];
          dest[address_index++] = result_buffer[3];
        }

        if (num_r_f > 128)
        { //129--256 
          dest[address_index++] = result_buffer[4];
          dest[address_index++] = result_buffer[5];
          dest[address_index++] = result_buffer[6];
          dest[address_index++] = result_buffer[7];
        }

        if (num_r_f > 256)
        { //257-511
          dest[address_index++] = result_buffer[8];
          dest[address_index++] = result_buffer[9];
          dest[address_index++] = result_buffer[10];
          dest[address_index++] = result_buffer[11];
          dest[address_index++] = result_buffer[12];
          dest[address_index++] = result_buffer[13];
          dest[address_index++] = result_buffer[14];
          dest[address_index++] = result_buffer[15];
        }
      }             
    }
}




//////////////////////////////////////////int//////////////////////////////////////////////////////////
struct LinearModelSample_int
{
  fp_type value;            //!< rating of this example
  DENSE_ONLY( hazy::vector::FVector<unsigned int> vector;); //!< feature vector

  LinearModelSample_int()
  { }


  LinearModelSample_int(
    fp_type val,
    unsigned int  *values,
    int *index, 
    unsigned len,
    int dimension
  ) : value(val) {

    hazy::vector::SVector<unsigned int> sparse_vector(values, index, len);

    unsigned int *zeros = new unsigned int[dimension];
    hazy::vector::FVector<unsigned int> *temp_vector = new hazy::vector::FVector<unsigned int>(zeros, dimension);

    // set to zero
    hazy::vector::Zero(*temp_vector);
    
    hazy::vector::ScaleAndAdd(
		      *temp_vector,
		      sparse_vector,
		      1.0
    );

//Converte from tmp_vector to dest_vector.
	unsigned int *dest_addr =  (unsigned int *)aligned_alloc(64, compute_num_CLs_per_sample(dimension)*16 * sizeof(unsigned int));
    //hazy::vector::FVector<unsigned int> *dest_vector = new hazy::vector::FVector<unsigned int>(zeros, dimension);
	bitweaving_on_each_sample(dest_addr, zeros, dimension);

    vector.size   = (*temp_vector).size;
    vector.values = dest_addr; //(*temp_vector).values;
    free(zeros);
  }


  LinearModelSample_int(const LinearModelSample_int &o) {
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
  
  void exchange_sample(LinearModelSample_int &exchange)
  {
    //exhange the values.
    unsigned int* this_vector = (unsigned int*)(this->vector.values);
    unsigned int* exch_vector = (unsigned int*)exchange.vector.values;
    unsigned this_size   = this->vector.size;
    unsigned exch_size   = exchange.vector.size;
    unsigned char temp[this_size];

	unsigned real_size = 16 * compute_num_CLs_per_sample(this_size);
		
    memcpy(         temp,            (void *) this_vector,   real_size * sizeof(int));
    memcpy((void *) this_vector,     (void *) exch_vector,   real_size * sizeof(int));
    memcpy((void *) exch_vector,          temp,              real_size * sizeof(int));

     //exhange the label.
    fp_type label_tmp;
    label_tmp        = this->value;
    this->value      = exchange.value;      
    exchange.value   = label_tmp;    
  }

};

#endif

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

  LinearModelSample_char(int dimension) {
	unsigned char *zeros = (unsigned char *)aligned_alloc(64, dimension * sizeof(char)); //new unsigned char[dimension];
	hazy::vector::FVector<unsigned char> *temp_vector = new hazy::vector::FVector<unsigned char>(zeros, dimension);

    vector.size   = (*temp_vector).size;
    vector.values = (*temp_vector).values;

  }


void regroup_from_bitweaving(LinearModelSample_int &src, unsigned num_bits)
  {
    //copy the label of this sample...
	this->value  = src.value;

    //copy the vectors of this sample...		
  	hazy::vector::FVector<unsigned  int> src_vector   = src.vector;
  	hazy::vector::FVector<unsigned char> dest_vector  = this->vector;
	
	hazy::vector::Convert_from_bitweaving (dest_vector, src_vector, num_bits);
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
