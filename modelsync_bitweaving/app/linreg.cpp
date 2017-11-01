#include "linearmodel_exec.h"
#include "linreg/linreg_exec.h"

int main(int args, char** argv)
{
  return LinearModelExec::Run<LinRegExec>(args, argv);
}
