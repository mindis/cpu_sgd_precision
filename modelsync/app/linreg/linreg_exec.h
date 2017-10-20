#ifndef _LIN_REG_EXEC_H
#define _LIN_REG_EXEC_H



#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
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
	  
		  static void CalcModelUpdate(LinearModelSample_char const * const &samples, size_t * current_batch, size_t actual_num_elements_in_batch, LinearModel *model, LinearModelParams const &params, unsigned tid)
		  {
			hazy::vector::FVector<fp_type> &x		= model->weights;
			hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];
	  
			//float scale = -model->batch_step_size/params.totalNumSamples;
			float scale = -model->batch_step_size; ///(float)params.batch_size;
		  
			for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
			{
			  //read the sample:
			  const LinearModelSample_char &sample = samples[current_batch[i]];
	  
	  //		float sample_float[sample.vector];
				/** Original Lin Reg Stuff */
				fp_type delta;
				delta = scale * (Dot( g_local, sample.vector) - sample.value);
	  
				// linear regression
				hazy::vector::ScaleAndAdd(
					g_local,
					sample.vector,
					delta
					);
			}
		  }
	  
		  static fp_type SingleLoss(const LinearModelSample_char &s, LinearModel *m)
		  {
			// determine how far off our model is for this example
			hazy::vector::FVector<fp_type> const &x = m->weights;
			fp_type dot = hazy::vector::Dot(x, s.vector);
	  
			// linear regression
			//printf("dot = %f, s.value = %f\n", dot, s.value);
			fp_type difference = (dot - s.value)/256.0;
			return 0.5 *difference * difference;// 
		  }
	  
		  static fp_type ComputeMetaLoss(const LinearModelSample_char &s, LinearModelParams const &params)
		  {
			return 0.0;
		  }
	  

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
          delta = scale * (Dot( g_local, sample.vector) - sample.value);

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
      hazy::vector::FVector<fp_type> &x       = model->weights;
      hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];

	  float scale = -model->batch_step_size; ///(float)params.batch_size;
	  
      for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
      {

        //read the sample:
        const LinearModelSample &sample = samples[current_batch[i]];

          /** Original Lin Reg Stuff */

          fp_type delta = scale * (Dot( g_local, sample.vector) - sample.value);

          // linear regression
          hazy::vector::ScaleAndAdd(
              g_local,
              sample.vector,
              delta
              );
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
