#include <inttypes.h>

#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/binfscan.h"
#include "hazy/types/entry.h"
#include "hazy/scan/memscan.h"
#include "hazy/scan/sampleblock.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/fvector.h"
#include "utils.h"

// Offset to modify rows and columns by, may be set to 1 by compiler
// to subtract 1 from each row and column to compensate for matlab files (only in TSV Files)
#ifndef OFFSET
#define OFFSET 0
#endif

typedef double fp_type;

struct Sample
{
  fp_type value;            //!< rating of this example
  hazy::vector::SVector<const fp_type> vector;

  Sample()
  { }

  Sample( fp_type val, fp_type const *values, int *index, unsigned len) : value(val), vector(values, index, len)
  { }

  Sample(const Sample &o) {
    value = o.value;
    vector.values = o.vector.values;
    vector.index = o.vector.index;
    vector.size = o.vector.size;
  }
};

template <class Scan> size_t LoadSamples(Scan &scan, hazy::vector::FVector<Sample> &ex)
{
  std::vector<Sample> examps;
  int lastrow = -1;
  double rating = 0.0;
  std::vector<fp_type> data;
  std::vector<int> index;

  int max_col = 0;
  int cnt = 0;

  while (scan.HasNext())
  {
    const hazy::types::Entry &e = scan.Next();
    if (lastrow == -1) {
      // this will be the case at the beginning
      lastrow = e.row;
    }
    if ((lastrow != e.row) || (!scan.HasNext())) {
      // finish off the previous vector and start a new one
      lastrow = e.row;

      if(!scan.HasNext()) {
        if (e.col < 0) {
          rating = e.rating;
        } else {
          if (e.col > max_col) {
            max_col = e.col;
          }
          data.push_back(e.rating);
          index.push_back(e.col);
        }
      }

      fp_type *d = new fp_type[data.size()];
      int *i = new int[data.size()];
      for (size_t j = 0; j < data.size(); j++) {
        d[j] = data[j];
        i[j] = index[j];
      }

      Sample temp(rating, d, i, data.size());
      examps.push_back(temp);
      rating = 0.0;
      data.clear();
      index.clear();

      cnt++;
    }

    if (e.col < 0) {
      rating = e.rating;
    } else {
      if (e.col > max_col) {
        max_col = e.col;
      }
      data.push_back(e.rating);
      index.push_back(e.col);
    }
  }

  // Copy from temp vector into persistent memory
  ex.size = examps.size();
  ex.values = new Sample[ex.size];
  for (size_t i = 0; i < ex.size; i++) {
    new (&ex.values[i]) Sample(examps[i]);
  }
  return max_col+1;
}

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    printf("usage: split_data SPLITS BINARY FILENAME\n");
    printf("  to split the FILENAME (.bin if BINARY else .tsv) to SPLITS files'\n");
    return 0;
  }
  assert(argc == 4);
  int splits = atoi(argv[1]);
  int binary = atoi(argv[2]);
  char* filename(argv[3]);

  printf("Splitting file '%s' in %i parts (binary: %i)\n", filename, splits, binary);

  // Generating names
  char*** newNames = new char**[splits];

  for(int i = 0; i < splits; ++i)
  {
    newNames[i] = new char*;
    getSplitNameOfFile(filename, i, splits, newNames[i]);

    printf("new name[%i]: %s\n", i, *newNames[i]);
  }

  // Read all the data
  hazy::vector::FVector<Sample> samps;
  int maxCol = 0;
  if(binary == 0)
  {
    if(OFFSET == 0)
    {
      hazy::scan::TSVFileScanner scan(filename);
      maxCol = LoadSamples(scan, samps);
    }
    else
    {
      hazy::scan::MatlabTSVFileScanner scan(filename);
      maxCol = LoadSamples(scan, samps);
    }
  }
  else
  {
    hazy::scan::BinaryFileScanner scan(filename);
    maxCol = LoadSamples(scan, samps);
  }
  printf("Max column (i.e. dimension of data): %i\n", maxCol);

  unsigned long tot = 0;
  for(uint64_t j = 0; j < samps.size; ++j)
  {
    const Sample &sample = samps[j];
    tot += sample.vector.size + 1;
  }
  printf("Spliting %lu samples (with %lu total entries)\n", samps.size, tot);

  long unsigned int counts[splits];
  long unsigned int elemsPerSplit = samps.size / splits;
  for(int i = 0; i < splits-1; ++i)
  {
    counts[i] = elemsPerSplit;
    printf("%i will have %lu elements\n", i, counts[i]);
  }
  counts[splits-1] = elemsPerSplit + (samps.size % splits);
  printf("%i will have %lu elements\n", splits-1, counts[splits-1]);
  

  FILE** files = new FILE*[splits];
  for(int i = 0; i < splits; ++i)
  {
    files[i] = fopen(*newNames[i], "w");
    if(!files[i])
    {
      perror("cannot open output file");
      return 0;
    }
  }

  // Process data
  //hazy::scan::MemoryScan< Sample > memScan(samps);
  hazy::scan::MemoryScanNoPermutation< Sample > memScan(samps);
  hazy::types::Entry e;
  while(memScan.HasNext()) // SHOULD ONLY BE CLASSED ONCE WITH MemoryScan
  {
    hazy::scan::SampleBlock<Sample> &block = memScan.Next();
    int curElement = 0;
    if(binary)
    {
      for(int i = 0; i < splits; ++i)
      {
        uint64_t tot = 0;
        for(uint64_t j = 0; j < counts[i]; ++j)
        {
          const Sample &sample = block.ex.values[block.perm[curElement]];
          tot += sample.vector.size + 1;
          curElement++;
        }
        printf("Split nr. %i - Found %lu examples (with %lu entries).\n", i, counts[i], tot);
        fwrite(&tot, sizeof(tot), 1, files[i]);
      }
    }

    curElement = 0;
    for(int i = 0; i < splits; ++i)
    {
      for(long unsigned int k = 0; k < counts[i]; ++k)
      {
        const Sample &sample = block.ex.values[block.perm[curElement]];

        if(binary)
        {
          e.row = block.perm[curElement];
          //e.row = k;
          e.col = -2;
          e.rating = sample.value;
          fwrite(&e, sizeof(e), 1, files[i]);
        }
        else
        {
          fprintf(files[i], "%lu\t%i\t%lf\n", block.perm[curElement]+OFFSET, (-2)+OFFSET, sample.value);
          //fprintf(files[i], "%lu\t%i\t%lf\n", k+OFFSET, (-2)+OFFSET, sample.value);
        }

        for(uint64_t j = 0; j < sample.vector.size; ++j)
        {
          if(binary)
          {
            e.col = sample.vector.index[j];
            e.rating = sample.vector.values[j];
            fwrite(&e, sizeof(e), 1, files[i]);
          }
          else
          {
            fprintf(files[i], "%lu\t%i\t%lf\n", block.perm[curElement]+OFFSET, sample.vector.index[j]+OFFSET, sample.vector.values[j]);
            //fprintf(files[i], "%lu\t%i\t%lf\n", k+OFFSET, sample.vector.index[j]+OFFSET, sample.vector.values[j]);
          }
        }

        curElement++;
      }
    }
  }

  // DELETE STUFF
  for(int i = 0; i < splits; ++i)
  {
    fclose(files[i]);
  }
  delete[] files;

  // delete new names
  for(int i = 0; i < splits; ++i)
  {
    delete *newNames[i];
    delete newNames[i];
  }
  delete[] newNames;

  return 0;
}
