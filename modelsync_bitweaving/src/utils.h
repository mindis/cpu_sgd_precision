#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>
#include <stdarg.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>

const double epsilon = 1e-8; // For float comparision
//const double epsilon = 1e-18; // For double comparision

void printDebug(int id, const char* format, ... )
{
#ifdef _DEBUG
  va_list args;
  char f[1024];
  va_start(args, format);
  vsprintf(f, format, args);
  va_end(args);
  printf("%i - %s\n", id, f);
#endif
}

bool fileExists(const char *filename)
{
  std::ifstream ifile(filename);
  return (bool)ifile;
}

void copyString(char* from, char** const to)
{
  std::string str(from);

  char* newStr = new char[str.length()+1];
  std::strcpy (newStr, str.c_str());

  *to = newStr;
}

void getSplitNameOfFile(char* filename, int split, int total, char** const newName)
{
  std::string sFilename(filename);
  std::string::size_type i = sFilename.rfind('.', sFilename.length());
  std::string ending;
  std::string prefix;
  std::string dot = ".";
  if (i != std::string::npos)
  {
    ending = sFilename.substr(i+1, sFilename.length() - (i+1));
    prefix = sFilename.substr(0, i);
  }
  else
  {
    ending = "";
    dot = "";
    prefix = filename;
  }

  std::stringstream fmt;
  fmt << prefix << "_" << split << "_" << total << dot << ending;

  std::string str = fmt.str();

  char* newStr = new char[str.length()+1];
  std::strcpy (newStr, str.c_str());

  *newName = newStr;
}

void changeFilename(char* filename, std::string suffix, std::string newEnding, char** const newName)
{
  std::string sFilename(filename);
  std::string::size_type i = sFilename.rfind('.', sFilename.length());
  std::string prefix;
  if (i != std::string::npos)
  {
    prefix = sFilename.substr(0, i);
  }
  else
  {
    prefix = filename;
  }

  std::stringstream fmt;
  fmt << prefix << suffix << "." << newEnding;

  std::string str = fmt.str();

  char* newStr = new char[str.length()+1];
  std::strcpy (newStr, str.c_str());

  *newName = newStr;
}

/*! Returns the starting index for a given thread for dividing up examples
 * Loop using for (size_t i = GetStartIndex; i < GetEndIndex(); i++)
 * \param total the total number of examples
 * \param tid the thread id [0, 1, 2, ...]
 * \param total number of threads
 * \return the start of this thread's "block" to work on
 */
inline size_t GetStartIndex(size_t total, unsigned tid, unsigned nthreads) {
  return (total / nthreads) * tid;
}

/*! Returns the ending index + 1 for the given thread to use
 * Loop using for (size_t i = GetStartIndex; i < GetEndIndex(); i++)
 * \param total the total number of examples
 * \param tid the thread id [0, 1, 2, ...]
 * \param total number of threads
 * \return last index to process PLUS ONE
 */
inline size_t GetEndIndex(size_t total, unsigned tid, unsigned nthreads) {
  size_t block_size = total / nthreads;
  size_t start = block_size * tid;
  if (nthreads == tid+1) {
    return total;
  }
  if (start + block_size > total) {
    return total;
  }
  return start + block_size;
}

#endif
