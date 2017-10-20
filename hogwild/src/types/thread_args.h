#ifndef _TYPES_THREAD_ARGS_H
#define _TYPES_THREAD_ARGS_H

#include "types/aligned_pointer.h"
#include "hazy/scan/sampleblock.h"
#include "hazy/vector/fvector.h"

template< class Model, class Params, class Sample > struct ThreadArgs
{
  Model* model_;
  Params* params_;
  hazy::scan::SampleBlock<Sample>* block_;
  size_t *current_batch_;
  size_t actual_num_elements_in_batch;
  int rank_;
  AlignedPointer<double> *losses_;
  AlignedPointer<hazy::util::Clock>* compute_times_;
  AlignedPointer<hazy::util::Clock>* communicate_times_;
};

#endif
