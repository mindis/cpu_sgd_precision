#ifndef _TYPES_ALIGNED_POINTER_H
#define _TYPES_ALIGNED_POINTER_H

template< typename type > struct AlignedPointer
{
  type* ptr;
  char pad[CACHE_LINE_SIZE - ((sizeof(ptr) % CACHE_LINE_SIZE))];

  AlignedPointer() : ptr(nullptr)
  { }
};

#endif
