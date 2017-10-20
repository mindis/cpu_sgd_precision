#include <inttypes.h>

#include "hazy/scan/tsvfscan.h"
#include "hazy/scan/binfscan.h"
#include "hazy/types/entry.h"
#include "hazy/scan/memscan.h"
#include "hazy/scan/sampleblock.h"
#include "hazy/vector/svector.h"
#include "hazy/vector/fvector.h"
#include "utils.h"

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
    printf("usage: generate_split_file.out SPLITS FILENAME\n");
    printf("  to generate a split file base on FILENAME (has to be in binary format)'\n");
    return 0;
  }
  assert(argc == 3);
  int splits = atoi(argv[1]);
  char* filename(argv[2]);
  
  // Generating names
  char** newName = new char*;

  std::stringstream s;
  s << "_seek_points_" << splits;
  changeFilename(filename, s.str(), "bin", newName);

  printf("Output filename: %s\n", *newName);

  // Read all the data
  hazy::vector::FVector<Sample> samps;
  hazy::scan::BinaryFileScanner scan(filename);
  int maxCol = LoadSamples(scan, samps);
  printf("Max column (i.e. dimension of data): %i\n", maxCol);

  FILE* file = fopen(*newName, "w");
  if(!file)
  {
    perror("cannot open output file");
    return 0;
  }

  uint64_t pos = sizeof(uint64_t);
  uint64_t prev = pos;
  uint64_t diff = 0;
  fwrite(&pos, sizeof(pos), 1, file);
  printf("%lu\t", pos);
  //fprintf(file, "%lu\t", pos);

  uint64_t size_of_chunks = samps.size / splits;

  for(uint64_t j = 0; j < samps.size; ++j)
  {
    const Sample &sample = samps[j];
    if (j != 0 && (j % size_of_chunks) == 0 && j + size_of_chunks <= samps.size)
    {
      diff = pos-prev;
      fwrite(&diff, sizeof(diff), 1, file);
      fwrite(&pos, sizeof(pos), 1, file);
      printf("%lu\n%lu\t", pos - prev, pos);
      //fprintf(file, "%lu\n%lu\t", pos - prev, pos);
      prev = pos;
    }
    pos += (sample.vector.size + 1) * sizeof(hazy::types::Entry);
  }
  diff = pos-prev;
  fwrite(&diff, sizeof(diff), 1, file);
  printf("%lu\n", pos - prev);
  //fprintf(file, "%lu\n", pos - prev);
  
  if (uint64_t(splits) > samps.size)
  {
    uint64_t zero = 0;
    for(unsigned int k = 0; k < splits - samps.size; ++k)
    {
      fwrite(&zero, sizeof(zero), 1, file);
      fwrite(&zero, sizeof(zero), 1, file);
      printf("%lu\t%lu\n", zero, zero);
    }
  }

  // DELETE STUFF
  fclose(file);
  delete newName;

  return 0;
}
