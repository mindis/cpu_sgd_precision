#ifndef _FRONTEND_UTIL_H
#define _FRONTEND_UTIL_H

#include <getopt.h>
#include <iostream>
#include <iomanip>

//! Extends the definition of option in getopt to add in a message.
struct extended_option
{
  const char *name;
  int has_arg,*flag,val;
  const char *msg;
};

//! convert extended options to regular options for use in getopt.
option* convert_extended_options(const extended_option* exs)
{
  using namespace std;
  int nOptions = 0; 
  while(exs[nOptions++].name != NULL);
  struct option* opts =  new option[nOptions];
  for(int i = 0; i < nOptions; i++) {
    opts[i].name    = exs[i].name;
    opts[i].has_arg = exs[i].has_arg;
    opts[i].flag    = exs[i].flag;
    opts[i].val     = exs[i].val;
  }
  return opts;
}


//! printout a usage string that lists all the options.
/*! extended options are all default options from above.
 * sysname is typically argv[0]
 * usage str contains the usage of the mandatory no flag options.
 */
void print_usage(const extended_option *exs, char* sysname, char *usage_str)
{
  using namespace std;
  string flags = "[";
  int i = 0;
  for(i=0;exs[i].name != NULL;i++) {
    string s = string("--") + string(exs[i].name);
    flags += (s + ((exs[i+1].name != NULL) ? string("|") : string("]")));
  }
  std::cout << "usage: " << sysname << " " << flags << " " << usage_str 
      << std::endl;
   

  for(i=0;exs[i].name != NULL;i++) {
     string s = string("--") + string(exs[i].name);
     std::cout << "\t" << setw(20) << left << s  << "\t: " << exs[i].msg 
         << std::endl;
   }
}

#endif
