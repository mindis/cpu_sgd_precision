#include "linearmodel_exec.h"
#include "logit/logit_exec.h"

int main(int args, char** argv)
{
  return LinearModelExec::Run<LogitExec>(args, argv);
}
