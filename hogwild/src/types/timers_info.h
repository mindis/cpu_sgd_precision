#ifndef _TYPES_TIMERS_INFO_H
#define _TYPES_TIMERS_INFO_H

#include "types/aligned_pointer.h"
#include "hazy/scan/sampleblock.h"
#include "hazy/vector/fvector.h"

struct TimersInfo
{
  double epochTime;
  double trainTime;
  double computeTime;
  double communicateTime;
  double testTime;

  TimersInfo() : epochTime(0.0), trainTime(0.0), computeTime(0.0), communicateTime(0.0), testTime(0.0)
  { }
};

#endif
