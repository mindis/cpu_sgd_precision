#include <assert.h>
#include <string.h>
#include <cstdlib>

typedef float fp_type;          //!< floating point type (double or float)
#define MPI_FP_TYPE MPI_FLOAT   //!< floating point MPI_Datatype (MPI_DOUBEL or MPI_FLOAT)
//typedef double fp_type;           //!< floating point type (double or float)
//#define MPI_FP_TYPE MPI_DOUBLE    //!< floating point MPI_Datatype (MPI_DOUBEL or MPI_FLOAT)

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifdef _DECREASING_STEPSIZES
#define DECREASING_STEPSIZES_ONLY(x) x;
#else
#define DECREASING_STEPSIZES_ONLY(x);
#endif

#ifdef _DENSE
#define DENSE_ONLY(x) x;
#else
#define DENSE_ONLY(x);
#endif

#ifdef _SPARSE
#define SPARSE_ONLY(x) x;
#else
#define SPARSE_ONLY(x);
#endif
