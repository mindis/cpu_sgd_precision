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

#include <cstdint>
#include <assert.h>     /* assert */
#include "pthread.h"

#include "cpu_mapping.h"
#include "huge_page.h"
#include "rand_tool.h"
#include "thread_tool.h"

#define NUM_VALUES 16L*1024*1024*1024


void memory_bandwidth_read(const float *data, float *bitmap, size_t tuples)
{   
    __m256 result_store   = _mm256_set1_ps(0.0);
    const float *data_end = &data[tuples];
    uint32_t *bitmap_word = (uint32_t*) bitmap;
    if (tuples) do {
        __m256 x1 = _mm256_load_ps( &data[0]);
        __m256 x2 = _mm256_load_ps( &data[8]);
        __m256 x3 = _mm256_load_ps( &data[16]);
        __m256 x4 = _mm256_load_ps( &data[24]);
        data += 32;
        __m256 c1 = _mm256_or_ps(x1, x2); //_mm256_add_ps
        __m256 c2 = _mm256_or_ps(x3, x4);
        __m256 c3 = _mm256_or_ps(c1, c2);
        result_store = _mm256_or_ps (result_store, c3);
    } while (data != data_end);
    
    _mm256_storeu_ps (bitmap, result_store); //make sure load instructions make sense...
}

void memory_bandwidth_read_2(const float *data, float *bitmap, size_t tuples)
{   
    __m256 result_store   = _mm256_set1_ps(0.0);
    const float *data_end = &data[tuples];
    uint32_t *bitmap_word = (uint32_t*) bitmap;
    if (tuples) do {
        __m256 x1 = _mm256_load_ps( &data[0]);
        __m256 x2 = _mm256_load_ps( &data[8]);
        data += 16;
        __m256 c3 = _mm256_or_ps(x1, x2); //_mm256_add_ps
        result_store = _mm256_or_ps (result_store, c3);
    } while (data != data_end);
    
    _mm256_storeu_ps (bitmap, result_store); //make sure load instructions make sense...
}

void memory_bandwidth_write(float *data, size_t tuples)
{   
    __m256 zero           = _mm256_set1_ps(0.0);
    const float *data_end = &data[tuples];
    //uint32_t *bitmap_word = (uint32_t*) bitmap;
    if (tuples) do {
         _mm256_store_ps (&data[0],  zero); 
         _mm256_store_ps (&data[8],  zero); 
         _mm256_store_ps (&data[16], zero); 
         _mm256_store_ps (&data[24], zero); 
         
        data += 32;
    } while (data != data_end);
}

void memory_bandwidth_write_2(float *data, size_t tuples)
{   
    __m256 zero           = _mm256_set1_ps(0.0);
    const float *data_end = &data[tuples];
    //uint32_t *bitmap_word = (uint32_t*) bitmap;
    if (tuples) do { 
         _mm256_stream_ps (&data[0],  zero); 
         _mm256_stream_ps (&data[8],  zero); 
         _mm256_stream_ps (&data[16], zero); 
         _mm256_stream_ps (&data[24], zero); 
         
        data += 32;
    } while (data != data_end);
}


typedef struct {
    int           model;
    unsigned int *addr;
    size_t        len;
    pthread_t     id;
    int           seed;
    int           thread;
    int           threads;
    uint64_t          *times[2];
    pthread_barrier_t *barrier;
} info_t;

void *run(void *arg)
{
    info_t *d                  = (info_t*) arg;

    int t;
    size_t i;
    pthread_barrier_t *barrier = d->barrier;
    int    model               = d->model;

    float *data                = (float *)d->addr; 
    size_t len                 = d->len;

    assert(len % 64 == 0);
    
    float *bitmap = (float *)malloc (1024);
#if 0
    //////////Log the elapsed time of each thread//////////
    pthread_barrier_wait(barrier++);
    uint64_t t1 = thread_time();

    if      (model == 0)
        memory_bandwidth_read(data, bitmap, len); //memory_bandwidth_read_2 
    else if (model == 1)
        memory_bandwidth_read_2(data, bitmap, len); //memory_bandwidth_read_2 
    else if (model == 2) 
        memory_bandwidth_write(data, len);
    else
        memory_bandwidth_write_2(data, len);

    //    
    t1 = thread_time() - t1;
    pthread_barrier_wait(barrier++);


    d->times[0][d->thread] = t1;


    pthread_barrier_wait(barrier++);
    if (d->thread == 0) 
    {
        uint64_t sum = 0;
        t1           = d->times[0][0];
        int t1_index = 0;
        
        for (t = 0; t < d->threads;  ++t) 
        {
            sum    += d->times[0][t];  //Compute the sum... 
            if (t1 < d->times[0][t])
            {
                t1       = d->times[0][t];
                t1_index = t;
            }   
        }
        printf("%d threads: %d-th thread requres_%6.1f ns, Slowest thread bandwidth: %5.3fGB/s, Average bandwidth: %5.3fGB/s\n",
               d->threads, t1_index, (double)t1, (double)(4*len)/(double)t1, (double)(4*len*d->threads)/(double)(sum/d->threads) );//
    }

    //printf("d->thread = %d, d->threads = %d\n", d->thread, d->threads);
    //printf("len = %lx\n", len);
#else
   for (int epoch = 0; epoch < 4; epoch++)
   { 
        

        uint64_t t1;    
        //////////Log the elapsed time of each thread//////////
        if (d->thread == 0)
        {   
            printf("epoch = %d\t", epoch);
            t1 = thread_time();
        }

        pthread_barrier_wait(barrier);

        if      (model == 0)
            memory_bandwidth_read(data, bitmap, len); //memory_bandwidth_read_2 
        else if (model == 1)
            memory_bandwidth_read_2(data, bitmap, len); //memory_bandwidth_read_2 
        else if (model == 2) 
            memory_bandwidth_write(data, len);
        else if (model == 3) 
            memory_bandwidth_write_2(data, len);

    //    
        pthread_barrier_wait(barrier);
        if (d->thread == 0)
            t1 = thread_time() - t1;
        //d->times[0][d->thread] = t1;
        pthread_barrier_wait(barrier);
        if (d->thread == 0) 
        {
        //uint64_t sum = 0;
        //t1           = d->times[0][0];
        //int t1_index = 0;
            printf("%d threads: time=%12.6f ns, len = %ld, Average bandwidth=%5.3fGB/s\n", 
                d->threads, (double)t1, len, (double)(4*len*d->threads)/( ((double)t1)*1.024*1.024*1.024 ) );//
        }
    }
#endif

    free(bitmap);
    //free();
    pthread_exit(NULL);
}

 
int main (int argc, char **argv)
{ 
    //////////////////Input parameters//////////////////
    int    threads      = argc > 1 ? atoi(argv[1]) : 1;
    size_t num_Gtuples  = argc > 2 ? atoi(argv[2]) : 1;
    int    disk_en      = argc > 3 ? atoi(argv[3]) : 0;
    int    model        = argc > 4 ? atoi(argv[4]) : 0;

    size_t tuples       = num_Gtuples * 1L * 1024L * 1024 * 1024;
    printf("Threads: %d, number of G tuples is %d, model is %d, disk_en = %d\n",threads, num_Gtuples, model, disk_en);

    size_t byte_len      = tuples * 4;
    unsigned int *p_addr;

    if (disk_en == 1)
    {
        //////////////////Open the file on the disk as fd///////////////////media/storage/zeke/data/data_64G_4k.dat
        int fd          = open ("../../../data/data_16G_4k.dat", O_RDWR);
        if (fd == -1) {
            printf("Error: cannot open the file data_16G_4k.dat\n");
            perror ("open");
            return 1;
        }
        printf("Succesfully open the file (fd)!!!\n");

        //////////////////Open the file on the disk//////////////////
        p_addr          =  (unsigned int *)mmap (0, byte_len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0); //MAP_POPULATE
        if (p_addr == MAP_FAILED) {
            perror ("mmap to the file (fd) error");
            return 1;
        }
        printf("Succesfully map the file to the memory location!!!\n");

        //if(madvise(p_addr,byte_len,MADV_SEQUENTIAL|MADV_WILLNEED)!=0) 
        //    printf("Couldn't set hints for p_addr\n");//cerr<<" Couldn't set hints for p_addr"<<endl;
    }   
    else
    {
        p_addr          =  (unsigned int *) malloc_huge_pages(byte_len);
       if (p_addr == NULL) {
            perror ("Huge malloc error\n");
            return 1;
        }

    }


    int num_barrier = 8;
    pthread_barrier_t barriers[num_barrier];
    for (int b = 0 ; b != num_barrier ; ++b)
        pthread_barrier_init(&barriers[b], NULL, threads);


    //////////////////Open the file on the disk//////////////////
    pthread_attr_t attr;
    cpu_set_t      set; //cpu_set_t *set = (cpu_set_t *) malloc (sizeof (cpu_set_t)); //
    pthread_attr_init(&attr);

    info_t info[threads];
    uint64_t times[2][threads];
    size_t num_tuples_per_thread  = ( (tuples/threads) & -64 ); // 
    printf("num_tuples_per_thread = 0x%lx\n", num_tuples_per_thread);

    for (int t = 0 ; t < threads; ++t) {

        int cpu_idx = get_cpu_id(t);
        CPU_ZERO(&set);
        CPU_SET(cpu_idx, &set);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &set);


        info[t].addr     = p_addr + (num_tuples_per_thread * t); //Address of 
        info[t].len      = num_tuples_per_thread; //(tuples / threads) & -64;

        info[t].seed     = rand();
        info[t].model    = model;
        info[t].thread   = t;
        info[t].threads  = threads;
        info[t].barrier  = barriers;
        info[t].times[0] = times[0];
        info[t].times[1] = times[1];
        pthread_create(&info[t].id, &attr, run, (void*) &info[t]);
    }
    
    for (int t = 0 ; t != threads ; ++t)
        pthread_join(info[t].id, NULL);


    if (0 != munmap (p_addr, byte_len)) 
        printf("Mumap error\n");

    for (int b = 0 ; b != num_barrier ; ++b)
        pthread_barrier_destroy(&barriers[b]);
    return EXIT_SUCCESS;
}
