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
  if (argc != 3)
  {
    printf("usage: random_reshuffle BINARY FILENAME\n");
    printf("  to reshuffle the entries in FILENAME (.bin if BINARY else .tsv) and generate a binary output file'\n");
    return 0;
  }
  assert(argc == 3);
  int binary = atoi(argv[1]);
  char* filename(argv[2]);

  // Generating names
  char** newName = new char*;

  changeFilename(filename, "_shuffled", "bin", newName);

  printf("Output filename: %s\n", *newName);

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

  FILE* file = fopen(*newName, "w");
  if(!file)
  {
    perror("cannot open output file");
    return 0;
  }

  // Process data
  hazy::scan::MemoryScan< Sample > memScan(samps);
  //hazy::scan::MemoryScanNoPermutation< Sample > memScan(samps);
  hazy::types::Entry e;
  while(memScan.HasNext()) // SHOULD ONLY BE CALLED ONCE WITH MemoryScan
  {
    hazy::scan::SampleBlock<Sample> &block = memScan.Next();
    uint64_t tot = 0;
    for(uint64_t j = 0; j < samps.size; ++j)
    {
      const Sample &sample = block.ex.values[block.perm[j]];
      tot += sample.vector.size + 1;
    }
    fwrite(&tot, sizeof(tot), 1, file);

    for(long unsigned int k = 0; k < samps.size; ++k)
    {
      const Sample &sample = block.ex.values[block.perm[k]];

      e.row = block.perm[k];
      //e.row = k;
      e.col = -2;
      e.rating = sample.value;
      fwrite(&e, sizeof(e), 1, file);

      for(uint64_t j = 0; j < sample.vector.size; ++j)
      {
        e.col = sample.vector.index[j];
        e.rating = sample.vector.values[j];
        fwrite(&e, sizeof(e), 1, file);
      }
    }
  }

  // DELETE STUFF
  fclose(file);
  delete newName;

  return 0;
}
