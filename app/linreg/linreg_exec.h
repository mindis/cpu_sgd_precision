#ifndef _LIN_REG_EXEC_H
#define _LIN_REG_EXEC_H


#include "hazy/vector/operations-inl.h"

#ifdef AVX2_EN
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#include "hazy/types/entry.h"

#define QUANTIZATION_FLAG_SAMPLE 1
#define QUANTIZATION_FLAG_GRADIENT 2
#define QUANTIZATION_FLAG_MODEL 4

class LinRegExec
{
  public:

    static void CalcModelUpdate(LinearModelSample_short const * const &samples, size_t * current_batch, size_t actual_num_elements_in_batch, LinearModel *model, LinearModelParams const &params, unsigned tid)
    {
      hazy::vector::FVector<fp_type> &x       = model->weights;
      hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];

      //float scale = -model->batch_step_size/params.totalNumSamples;
      float scale = -model->batch_step_size; ///(float)params.batch_size;
    
      for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
      {
        //read the sample:
        const LinearModelSample_short &sample = samples[current_batch[i]];

//        float sample_float[sample.vector];
          /** Original Lin Reg Stuff */
          fp_type delta;
          delta = scale * (Dot( x, sample.vector) - sample.value);

          // linear regression
          hazy::vector::ScaleAndAdd(
              g_local,
              sample.vector,
              delta
              );
      }
    }

    static fp_type SingleLoss(const LinearModelSample_short &s, LinearModel *m)
    {
      // determine how far off our model is for this example
      hazy::vector::FVector<fp_type> const &x = m->weights;
      fp_type dot = hazy::vector::Dot(x, s.vector);

      // linear regression
      //printf("dot = %f, s.value = %f\n", dot, s.value);
      fp_type difference = (dot - s.value)/65536.0;
      return 0.5 *difference * difference;// 
    }

    static fp_type ComputeMetaLoss(const LinearModelSample_short &s, LinearModelParams const &params)
    {
      return 0.0;
    }


    static void CalcModelUpdate(LinearModelSample const * const &samples, size_t * current_batch, size_t actual_num_elements_in_batch, LinearModel *model, LinearModelParams const &params, unsigned tid)
    {
      hazy::vector::FVector<fp_type> &x = model->weights;
      hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];

      hazy::vector::FVector<fp_type> quantized_model;
      if((params.quantization & QUANTIZATION_FLAG_MODEL) == QUANTIZATION_FLAG_MODEL)
      {
        quantized_model.size = model->weights.size;
        quantized_model.values = new fp_type[ model->weights.size ];
      }

      //TODO Change this into sparse representation if _SPARSE is set
      hazy::vector::FVector<fp_type> quantized_sample_1;
      hazy::vector::FVector<fp_type> quantized_sample_2;
      if((params.quantization & QUANTIZATION_FLAG_SAMPLE) == QUANTIZATION_FLAG_SAMPLE)
      {
        quantized_sample_1.size = model->weights.size;
        quantized_sample_1.values = new fp_type[ model->weights.size ];
        quantized_sample_2.size = model->weights.size;
        quantized_sample_2.values = new fp_type[ model->weights.size ];
      }

      //The quantization level
      unsigned qlevel = params.quantizationLevel;

      //float scale = -model->batch_step_size/params.totalNumSamples;
		float scale = -model->batch_step_size; ///(float)params.batch_size;
	  
      for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
      {

        //read the sample:
        const LinearModelSample &sample = samples[current_batch[i]];

        if((params.quantization & QUANTIZATION_FLAG_MODEL) == QUANTIZATION_FLAG_MODEL)
        {
          /** MODEL QUANTIZATION */ 
          //        	zero the quantized model
          hazy::vector::Zero(quantized_model);
          // // 		  copy it in 
          hazy::vector::ScaleAndAdd( quantized_model, x, 1 );
          // // 		  quantize in place
          hazy::vector::QSGDQuantizeInto( quantized_model,  qlevel );
        }

        if((params.quantization & QUANTIZATION_FLAG_SAMPLE) == QUANTIZATION_FLAG_SAMPLE)
        {
          /** SAMPLE QUANTIZATION */ 

          //       //quantize the sample twice and copy it in
          hazy::vector::QSGDQuantizeOut( quantized_sample_1, sample.vector, qlevel );
          hazy::vector::QSGDQuantizeOut( quantized_sample_2, sample.vector, qlevel );
          //       
          /** COMPUTE GRADIENT VALUE */ 

          fp_type dot_1;
          fp_type dot_2;
          if((params.quantization & QUANTIZATION_FLAG_MODEL) == QUANTIZATION_FLAG_MODEL)
          {
            //fp_type delta_qsample = 0;
            dot_1 = scale * (Dot( quantized_sample_1, quantized_model ) - sample.value);
            dot_2 = scale * (Dot( quantized_sample_2, quantized_model ) - sample.value);
          }
          else
          {
            dot_1 = scale * (Dot( quantized_sample_1, x) - sample.value);
            dot_2 = scale * (Dot( quantized_sample_2, x) - sample.value);
          }

          //get the first sample to its proper value
          hazy::vector::Scale( quantized_sample_1, 0.5 * dot_2 );

          //quick hack to get the second sample to its proper value, and add it into the second
          hazy::vector::ScaleAndAdd( quantized_sample_1, quantized_sample_2, 0.5 * dot_1);

          // at this point quantized_sample_1 contains the approximate gradient 
          // computed on top of the two samples

          hazy::vector::ScaleAndAdd( g_local, quantized_sample_1, 1);
        }
        else
        {
          /** Original Lin Reg Stuff */

          fp_type delta;
          if((params.quantization & QUANTIZATION_FLAG_MODEL) == QUANTIZATION_FLAG_MODEL)
          {
            // calc change of vector
            delta = scale * (Dot( quantized_model, sample.vector) - sample.value);
          }
          else
          {
            delta = scale * (Dot( x, sample.vector) - sample.value);
          }

          // linear regression
          hazy::vector::ScaleAndAdd(
              g_local,
              sample.vector,
              delta
              );
        }
      }


      if((params.quantization & QUANTIZATION_FLAG_GRADIENT ) == QUANTIZATION_FLAG_GRADIENT )
      {
        /*** GRADIENT QUANTIZATION */ 
        hazy::vector::QSGDQuantizeInto( g_local, qlevel );
      }

      // Do proper delete of the allocated arrays
      if((params.quantization & QUANTIZATION_FLAG_MODEL) == QUANTIZATION_FLAG_MODEL)
      {
        delete[] quantized_model.values;
      }
      if((params.quantization & QUANTIZATION_FLAG_SAMPLE) == QUANTIZATION_FLAG_SAMPLE)
      {
        delete[] quantized_sample_1.values;
        delete[] quantized_sample_2.values;
      }
    }

    static fp_type SingleLoss(const LinearModelSample &s, LinearModel *m)
    {
      // determine how far off our model is for this example
      hazy::vector::FVector<fp_type> const &x = m->weights;
      fp_type dot = hazy::vector::Dot(x, s.vector);

      // linear regression
      fp_type difference = dot - s.value;
      return 0.5 * difference * difference;
    }

    static fp_type ComputeMetaLoss(const LinearModelSample &s, LinearModelParams const &params)
    {
      // determine how far off our model is for this example
      hazy::vector::FVector<fp_type> const &x = *params.x_hat;
      fp_type dot = hazy::vector::Dot(x, s.vector);

      // linear regression
      fp_type difference = dot - s.value;
      return 0.5 * difference * difference;
    }
};

#endif
