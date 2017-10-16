#ifndef _SVM_EXEC_H
#define _SVM_EXEC_H

#include "hazy/vector/operations-inl.h"

#ifdef AVX2_EN
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#include "hazy/types/entry.h"


class SvmExec
{
public:
  static void CalcModelUpdate(LinearModelSample const * const &samples, size_t * current_batch, size_t actual_num_elements_in_batch, LinearModel *model, LinearModelParams const &params, unsigned tid)
  {
    hazy::vector::FVector<fp_type> &x = model->weights;
    hazy::vector::FVector<fp_type> &g_local = model->local_gradients[tid];

    float scale = model->batch_step_size/params.totalNumSamples;

    for (unsigned i = 0; i < actual_num_elements_in_batch; i++)
    {
      const LinearModelSample &sample = samples[current_batch[i]];
      fp_type dot = hazy::vector::Dot(x, sample.vector);

      // SVM Stuff (hinge loss)
      
      if(sample.value*dot < 1) {
        hazy::vector::ScaleAndAdd(g_local, sample.vector, sample.value*scale);
      }
    }
  }

  static fp_type SingleLoss(const LinearModelSample &s, LinearModel *m)
  {
    // determine how far off our model is for this example
    hazy::vector::FVector<fp_type> const &x = m->weights;
    fp_type dot = hazy::vector::Dot(x, s.vector);

    fp_type val = s.value*dot;
    // hinge loss
    if(val > 1)
      return 0;
    
    return 1 - val;
  }

  static fp_type ComputeMetaLoss(const LinearModelSample &s, LinearModelParams const &params)
  {
    // determine how far off our model is for this example
    hazy::vector::FVector<fp_type> const &x = *params.x_hat;
    fp_type dot = hazy::vector::Dot(x, s.vector);

    fp_type val = s.value*dot;
    // hinge loss
    if(val > 1)
      return 0;
    
    return 1 - val;
  }
};

#endif
