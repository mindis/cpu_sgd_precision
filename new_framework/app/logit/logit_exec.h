#ifndef _LOGIT_EXEC_H
#define _LOGIT_EXEC_H

#include "hazy/vector/operations-inl.h"

#ifdef AVX2_EN
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#include "hazy/types/entry.h"

class LogitExec
{
public:
  static void CalcModelUpdate(LinearModelSample const * const &samples, size_t * current_batch, size_t actual_num_elements_in_batch, LinearModel *model, LinearModelParams const &params, unsigned tid)
  {
    hazy::vector::FVector<fp_type> &x = model->weights;
    hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];

    //TODO Why are we not deviding by the toal number of training samples?!?
    float scale = model->batch_step_size/params.totalNumSamples;
    //float scale = model->batch_step_size;

    for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
    {
      const LinearModelSample &sample = samples[current_batch[i]];

      // Logistic Regression Stuff

      // calc change of vector
      fp_type delta = scale * sample.value / (1 + exp(sample.value * Dot(x, sample.vector)));

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

    // logistic regression
    return log(1 + exp(-s.value * dot));
  }

  static fp_type ComputeMetaLoss(const LinearModelSample &s, LinearModelParams const &params)
  {
    // determine how far off our model is for this example
    hazy::vector::FVector<fp_type> const &x = *params.x_hat;
    fp_type dot = hazy::vector::Dot(x, s.vector);

    // logistic regression
    return log(1 + exp(-s.value * dot));
  }
};

#endif
