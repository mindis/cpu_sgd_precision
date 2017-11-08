#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#ifdef __linux__
    #include <malloc.h>
#endif

#define AVX2_EN

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#ifdef AVX2_EN
#include "hazy/vector/operations-inl_avx2.h"
#include "hazy/vector/dot-inl_avx2.h"
#include "hazy/vector/scale_add-inl_avx2.h"
#else
#include "hazy/vector/operations-inl.h"
#include "hazy/vector/dot-inl.h"
#include "hazy/vector/scale_add-inl.h"
#endif

#define NUM_VALUES 47236

int test_mmap()
{
    printf("===============================================================\n");
    printf("========================== Test mmap===========================\n");
    printf("===============================================================\n");
    struct stat sb;
    off_t len;
    unsigned int *p;
    int fd;

    fd = open ("test.dat", O_RDWR);
    if (fd == -1) {
      printf("Error: cannot open the file test.dat\n");
      perror ("open");
      return 1;
    }
/*
        if (fstat(fd, &sb) == -1) { // == −1
                perror ("fstat");
                return 1;
        }

        if (!S_ISREG(sb.st_mode)) {
                fprintf (stderr, " is not a file\n");
                return 1;
        }
*/
    p = (unsigned int *)mmap (0, 10*1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
          perror ("mmap");
          return 1;
    }

    for (int i = 0; i < 2*1024*1024; i++)
      p[i] = i;


    for (int i = 1024*1024; i < 1024*1024 + 10; i++)
      printf("%d: %d\n", i, p[i]);

    return 1;
}


void main ()
{ 
  test_mmap();

}
