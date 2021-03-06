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

#include "cpu_mapping.h"
#include "huge_page.h"
#include "rand_tool.h"


#define NUM_VALUES 16L*1024*1024*1024

int test_mmap()
{
    printf("===============================================================\n");
    printf("========================== Test mmap===========================\n");
    printf("===============================================================\n");
    struct stat sb;
    off_t len;
    unsigned int *p;
    int fd;

    fd = open ("../../../data/data_16G_4k.dat", O_RDWR);
    if (fd == -1) {
      printf("Error: cannot open the file data_16G_4k.dat\n");
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
    printf("Succesfully open the file!!!\n");

    p =  (unsigned int *)mmap (0, NUM_VALUES, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
          perror ("mmap error");
          return 1;
    }
    printf("Succesfully map the file to the memory location!!!\n");

/*
    for (uint64_t i = 0; i < NUM_VALUES/4; i++) //*1024*1024
      p[i] = (i&0xffffffff);
*/
    printf("Succesfully wrting the data to the file!!!\n");

    uint64_t sum = 0;
    for (uint64_t i = 0; i < NUM_VALUES/4; i++)
    {
      sum += p[i];
    }
    
    printf("Sum = 0x%x\n", sum);

    return 1;
}


void main ()
{ 
  test_mmap();

}
