#ifndef SPARSE_VECTOR_H
#define SPARSE_VECTOR_H

#include <vector>
#include <utility>
#include <iostream>
#include <assert.h>
#include <cmath>

using namespace std;

typedef unsigned int indices_t;
typedef pair<indices_t, fp_type> sparse_item;
typedef vector<sparse_item> sparse_vector;

void make_dense(const sparse_vector &sparse_vector, unsigned int dimension, vector<fp_type> &dense_vector)
{
  assert(dense_vector.size() == dimension);
  for (const pair<unsigned int, fp_type>& t : sparse_vector) {
    assert(t.first < dimension);
    dense_vector[t.first] = t.second;
  }
}

void make_dense(const sparse_vector &sparse_vector, unsigned int dimension, hazy::vector::FVector<fp_type> &dense_vector)
{
  assert(dense_vector.size == dimension);
  for (const pair<unsigned int, fp_type>& t : sparse_vector) {
    assert(t.first < dimension);
    dense_vector[t.first] = t.second;
  }
}

void make_sparse(const vector<fp_type> &dense_vector, sparse_vector &sparse_vector)
{
  assert(sparse_vector.size() == 0);
  for (unsigned int i = 0; i < dense_vector.size(); i++) {
    fp_type value = dense_vector[i];
    if(std::abs(value) >= epsilon) {
      sparse_vector.push_back(make_pair(i, value));
    }
  }
}

void make_sparse(const hazy::vector::FVector<fp_type> &dense_vector, sparse_vector &sparse_vector)
{
  assert(sparse_vector.size() == 0);
  for (unsigned int i = 0; i < dense_vector.size; i++) {
    fp_type value = dense_vector.values[i];
    if(std::abs(value) >= epsilon) {
      sparse_vector.push_back(make_pair(i, value));
    }
  }
}

void sum_sparse_intofirst(sparse_vector &first, const sparse_vector &second, unsigned int &count) {
    unsigned int p1 = 0;
    unsigned int p2 = 0;

    count = 0;

    // Create copy of first (could be avoided by making use of shared_ptr
    sparse_vector tmp(first);

    // Sum sparse vector
    sparse_vector& result = first;
    result.clear();
    result.reserve(tmp.size() + second.size());
    while(p1 < tmp.size() || p2 < second.size()) {
      //cout << "P1: " << p1 << " || P2: " << p2;
      //cout << " I1: " << tmp[p1].first << " || I2: " << second[p2].first;
      //cout << " V1: " << tmp[p1].second << " || V2: " << second[p2].second << endl;
      if((p1 == tmp.size()) || (p2 != second.size() && (second[p2].first < tmp[p1].first))) {
        result.push_back(second[p2]);
        p2++;
      } else if((p2 == second.size()) || (tmp[p1].first < second[p2].first)) {
        result.push_back(tmp[p1]);
        p1++;
      } else {
        // index of receiver as index of sender must be equal
        assert(tmp[p1].first == second[p2].first);
        fp_type sum = tmp[p1].second + second[p2].second;
        if(std::abs(sum) >= epsilon) {
          result.push_back(make_pair(tmp[p1].first, sum));
        }
        p1++;
        p2++;
      }
      count++;
    }

    result.shrink_to_fit(); 
}

void sum_sparse_intofirst(sparse_vector &first, const sparse_vector &second) {
  unsigned int count = 0;
  sum_sparse_intofirst(first, second, count);
}

void print_sparse_vector(const sparse_vector &sparse_vector) {
  for(unsigned int i = 0; i < sparse_vector.size(); i++) {
    auto pair = sparse_vector[i];
    cout << "(" << pair.first << "," << pair.second << ")" << endl;
  }
}

bool spare_vector_equal(const sparse_vector &v1, const sparse_vector &v2) {
  if(v1.size() != v2.size())
    return false;

  for(unsigned int i = 0; i < v1.size(); ++i) {
    if(v1[i].first != v2[i].first)
      return false;

    if(std::abs(v1[i].second - v2[i].second) >= epsilon)
      return false;
  }

  return true;
}

#endif
