#ifndef _LINEARMODEL_LOADER_H
#define _LINEARMODEL_LOADER_H

#include "hazy/vector/dot-inl.h"
#include "hazy/vector/operations-inl.h"
#include "hazy/types/entry.h"

class LinearModelLoader
{
public:
  template <class Scan> static size_t LoadSamples(Scan &scan, hazy::vector::FVector<LinearModelSample> &ex, int dimension)
  {
    std::vector<LinearModelSample> examps;
    int lastrow = -1;
    double rating = 0.0;
    std::vector<fp_type> data;
    std::vector<int> index;

    int max_col = 0;

    while (scan.HasNext()) {
      const hazy::types::Entry &e = scan.Next();
      if (lastrow == -1) {
        // this will be the case at the beginning
        lastrow = e.row;
      }
      if ((lastrow != e.row) || (!scan.HasNext())) {
        // finish off the previous vector and start a new one
        lastrow = e.row;

        if(!scan.HasNext()) {
          data.push_back(e.rating);
          index.push_back(e.col);
        }

        fp_type *d = new fp_type[data.size()];
        int *i = new int[data.size()];
        for (size_t j = 0; j < data.size(); j++) {
          d[j] = data[j];
          i[j] = index[j];
        }
        LinearModelSample temp(rating, d, i, data.size(), dimension);
        examps.push_back(temp);
        rating = 0.0;
        data.clear();
        index.clear();
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
    ex.values = new LinearModelSample[ex.size];
    for (size_t i = 0; i < ex.size; i++) {
      new (&ex.values[i]) LinearModelSample(examps[i]);
    }
    return max_col+1;
  }


  template <class Scan> static size_t LoadSamples(Scan &scan, hazy::vector::FVector<LinearModelSample_int> &ex, int dimension)
  {
    std::vector<LinearModelSample_int> examps;
    int lastrow = -1;
    double rating = 0.0;
    std::vector<unsigned int> data;
    std::vector<int> index;
/*
	unsigned int *tmp_data; //[dimension]
	int inc_counter = 0;
	tmp_data = (unsigned int *) malloc (dimension * sizeof(unsigned));
	for (int kk = 0; kk < dimension; kk++)
		tmp_data[kk] = 0;
*/	
    int max_col = 0;

    while (scan.HasNext()) {
      const hazy::types::Entry &e = scan.Next();
      if (lastrow == -1) {
        // this will be the case at the beginning
        lastrow = e.row;
      }
      if ((lastrow != e.row) || (!scan.HasNext())) {
        // finish off the previous vector and start a new one
        lastrow = e.row;

        if(!scan.HasNext()) {
           data.push_back((unsigned int)(e.rating * 4294967295.0));        //e.rating 65535
          index.push_back(e.col);
        }

        unsigned int *d = new unsigned int[data.size()];
                   int *ii = new int[data.size()];
        for (size_t j = 0; j < data.size(); j++) {
          d[j] = data[j];
          ii[j] = index[j];
/*		  
		  if (inc_counter == 1)
		  {
			 printf("%d: 0x%8x\n", ii[j], data[j]);//
			 tmp_data[ii[j]] =  data[j]; //printf("%d:%x  ", kk, (ex.values[i].vector)[kk] );
		  } 	
*/		  
        }

		//inc_counter++;
		
        LinearModelSample_int temp(rating, d, ii, data.size(), dimension);
        examps.push_back(temp);
        rating = 0.0;
        data.clear();
        index.clear();
		free(d); 
		free(ii); 
      }

      if (e.col < 0) {
        rating = e.rating;
      } else {
        if (e.col > max_col) {
          max_col = e.col;
        }
        data.push_back( (unsigned int)(e.rating * 4294967295.0) ); //e.rating
        index.push_back( e.col );

		//printf("%f, 0x%8x  ", e.rating ,(unsigned int)(e.rating * 4294967295.0));
	  //  { for (int kk=0; kk < sample_char.vector.size; kk++) //
      //    if ((sample_char.vector)[kk] != 0)
      //  	printf("%d:%x  ", kk, (sample_char.vector)[kk] );
      //  }
      }
    }

    // Copy from temp vector into persistent memory
    ex.size   = examps.size();
    ex.values = new LinearModelSample_int[ex.size];
    for (size_t i = 0; i < ex.size; i++) {
      new (&ex.values[i]) LinearModelSample_int(examps[i]); //ex.values[i] = examps[i]; //
    }

/*
	printf("After loading: addr of ex[1].data is 0x%x\n", ex[1].vector.values);
    unsigned char dest[dimension];
    hazy::vector::FVector<unsigned char> dest_char_vector (dest, dimension);
    //hazy::vector::FVector<unsigned int> src_int_vector (dest_addr, dimension);
    hazy::vector::Convert_from_bitweaving(dest_char_vector, ex[1].vector, 8); //ex.values[1]
   
    for (int i = 0; i < dimension; i++)
        if ( (tmp_data[i]>>24) != dest_char_vector[i])
        {
            printf("Original_%d: src_0x%8x, dest_0x%x\n", i, tmp_data[i], dest_char_vector[i]);
            break;
        }	
*/		
	// for (int kk=0; kk < ex.values[1].vector.size; kk++) //
	//  if ((ex.values[1].vector)[kk] != 0)
	//	printf("%d:%x  ", kk, (ex.values[1].vector)[kk] );

	
    return max_col+1;
  }  


  template <class Scan> static size_t LoadSamples(Scan &scan, hazy::vector::FVector<LinearModelSample_short> &ex, int dimension)
  {
    std::vector<LinearModelSample_short> examps;
    int lastrow = -1;
    double rating = 0.0;
    std::vector<unsigned short> data;
    std::vector<int> index;

    int max_col = 0;

    while (scan.HasNext()) {
      const hazy::types::Entry &e = scan.Next();
      if (lastrow == -1) {
        // this will be the case at the beginning
        lastrow = e.row;
      }
      if ((lastrow != e.row) || (!scan.HasNext())) {
        // finish off the previous vector and start a new one
        lastrow = e.row;

        if(!scan.HasNext()) {
          data.push_back((unsigned short)(e.rating * 65535.9));        //e.rating
          index.push_back(e.col);
        }

        unsigned short *d = new unsigned short[data.size()];
                   int *i = new int[data.size()];
        for (size_t j = 0; j < data.size(); j++) {
          d[j] = data[j];
          i[j] = index[j];
        }
        LinearModelSample_short temp(rating, d, i, data.size(), dimension);
        examps.push_back(temp);
        rating = 0.0;
        data.clear();
        index.clear();
      }

      if (e.col < 0) {
        rating = e.rating;
      } else {
        if (e.col > max_col) {
          max_col = e.col;
        }
        data.push_back( (unsigned short)(e.rating * 65535.9) ); //e.rating
        index.push_back( e.col );
      }
    }

    // Copy from temp vector into persistent memory
    ex.size = examps.size();
    ex.values = new LinearModelSample_short[ex.size];
    for (size_t i = 0; i < ex.size; i++) {
      new (&ex.values[i]) LinearModelSample_short(examps[i]);
    }
    return max_col+1;
  }  


template <class Scan> static size_t LoadSamples(Scan &scan, hazy::vector::FVector<LinearModelSample_char> &ex, int dimension)
{
  std::vector<LinearModelSample_char> examps;
  int lastrow = -1;
  double rating = 0.0;
  std::vector<unsigned char> data;
  std::vector<int> index;

  int max_col = 0;

  while (scan.HasNext()) {
	const hazy::types::Entry &e = scan.Next();
	if (lastrow == -1) {
	  // this will be the case at the beginning
	  lastrow = e.row;
	}
	if ((lastrow != e.row) || (!scan.HasNext())) {
	  // finish off the previous vector and start a new one
	  lastrow = e.row;

	  if(!scan.HasNext()) {
		data.push_back((unsigned char)(e.rating * 255.9));		 //e.rating 65535
		index.push_back(e.col);
	  }

	  unsigned char *d = new unsigned char[data.size()];
				 int *i = new int[data.size()];
	  for (size_t j = 0; j < data.size(); j++) {
		d[j] = data[j];
		i[j] = index[j];
	  }
	  LinearModelSample_char temp(rating, d, i, data.size(), dimension);
	  examps.push_back(temp);
	  rating = 0.0;
	  data.clear();
	  index.clear();
	}

	if (e.col < 0) {
	  rating = e.rating;
	} else {
	  if (e.col > max_col) {
		max_col = e.col;
	  }
	  data.push_back( (unsigned char)(e.rating * 255.9) ); //e.rating
	  index.push_back( e.col );
	}
  }

  // Copy from temp vector into persistent memory
  ex.size = examps.size();
  ex.values = new LinearModelSample_char[ex.size];
  for (size_t i = 0; i < ex.size; i++) {
	new (&ex.values[i]) LinearModelSample_char(examps[i]);
  }
  return max_col+1;
}  

};

#endif
