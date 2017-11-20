#include "linearmodel_exec.h"
#include "svm/svm_exec.h"

int main(int args, char** argv)
{
  return LinearModelExec::Run<SvmExec>(args, argv);
}
