#ifndef MPI_FSCAN_H
#define MPI_FSCAN_H

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <inttypes.h>
#include <stddef.h> 

#include "hazy/types/entry.h"
#include <string.h>
#include <assert.h>

class MpiFileScanner {
  public:
    MpiFileScanner(const char *fname, uint64_t seek, uint64_t total) : posn_(0), size_(0), max_col_(0), array_(NULL), seek_(seek), total_(total/sizeof(hazy::types::Entry)) {

      // Initialize custom MPI_Datatype
      const int nitems = 3;
      int blocklengths[nitems] = {1, 1, 1};
      MPI_Datatype types[nitems] = {MPI_INT, MPI_INT, MPI_DOUBLE};
      MPI_Aint offsets[nitems];

      offsets[0] = offsetof(hazy::types::Entry, row);
      offsets[1] = offsetof(hazy::types::Entry, col);
      offsets[2] = offsetof(hazy::types::Entry, rating);

      MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_type_);
      MPI_Type_commit(&mpi_type_);

      if(MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_) != MPI_SUCCESS) {
        char buf[1024];
        sprintf(buf, "fopen failed for %s", fname);
        perror(buf);
        MPI_Abort(MPI_COMM_WORLD, 4);
      }
      buf_size_ = 1024 * 1024;
      array_ = new hazy::types::Entry[buf_size_];
      Reset();
    }

    ~MpiFileScanner() {
      MPI_File_close(&fh_);
      MPI_Type_free(&mpi_type_);
    }

    inline bool HasNext() const {
      return total_ > 0;
    }

    inline const hazy::types::Entry& Next() {
      const hazy::types::Entry &nxt = Peek();
      posn_++;
      total_--;
      max_col_ = std::max(max_col_, nxt.col);
      return nxt;
    }

    inline const hazy::types::Entry& Peek() {
      using hazy::types::Entry;
      assert(HasNext());
      if (posn_ < size_) {
        return array_[posn_];
      }
      // page in from file, this may not fill our buffer completely
      size_t toread = std::min(buf_size_, total_);
      assert(toread > 0);
      MPI_File_read( fh_, array_, toread, mpi_type_, MPI_STATUS_IGNORE );
      size_ = toread;
      // clear our buffer
      posn_ = 0;
      return array_[posn_];
    }

    inline int MaxColumn() const { return max_col_; }

    void Reset() {
      using hazy::types::Entry;
      MPI_File_seek(fh_, seek_, MPI_SEEK_SET );
      size_ = 0;
      posn_ = 0;
    }

  private:
    MPI_File fh_;
    MPI_Datatype mpi_type_;
    size_t posn_;
    size_t size_;
    int max_col_;
    hazy::types::Entry* array_;
    uint64_t seek_;
    uint64_t total_;
    size_t buf_size_;
};

#endif
