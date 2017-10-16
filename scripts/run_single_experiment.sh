# INFO:
# Script to run a single experiment
usage_message='Usage: sh scripts/run_single_experiment.sh $dataset(synthetic|higgs|rcv1|webspam_uni|webspam_tri) $strategy(PARAMETER_SERVER_SYNC|PARAMETER_SERVER_ASYNC|PARAMETER_SERVER_ASYNC_SPARSE|ALL_REDUCE|ALL_REDUCE_SPARSE|HOGWILD) $stepsize(EXPBACKOFF|DECREASING) $representation(SPARSE|DENSE) $nprocesses $nthreads $nepochs $batchsize (OPT:$lasso_regularizer $quantization $qlevel)'

# Arguments to be specified
dataset=$1
strategy=$2
stepsize=$3
representation=$4
nprocesses=$5
nthreads=$6
nepochs=$7
batchsize=$8
lasso_regularizerArg=$9
quantizationArg=${10}
qlevelArg=${11}

# For now fix (maybe also configurable in the future
beta=0.55
stepdecay=0.95

quantization=0
qlevel=1
lasso_regularizer=0

mpi_exec=mpirun

if [ -z "$dataset" ] || [ -z "$strategy" ] || [ -z "$stepsize" ] || [ -z "$representation"  ] || [ -z "$nprocesses" ] || [ -z "$nthreads" ] || [ -z "$nepochs" ] || [ -z "$batchsize" ] || [ "$nprocesses" -lt 1 ] || [ "$nthreads" -lt 1 ] || [ "$nepochs" -lt 1 ] || [ "$batchsize" -lt 0 ]
then
	echo $usage_message
	exit
fi

if [ ! -z "$quantizationArg" ]
then
quantization=$quantizationArg
fi

if [ ! -z "$qlevelArg" ]
then
qlevel=$qlevelArg
fi

if [ ! -z "$lasso_regularizerArg" ]
then
lasso_regularizer=$lasso_regularizerArg
fi

execs_folder=bin

if [ ${dataset,,} == synthetic ]
then
  trainfile=./data/synth[m=10000][d=1024][sigma=1][sparsity_fraction=0.5][mu_a=0][sigma_a=1]_sparse_shuffled.bin
  testfile=./data/synth[m=10000][d=1024][sigma=1][sparsity_fraction=0.5][mu_a=0][sigma_a=1]_sparse_test_shuffled.bin
  metafile=./data/synth[m=10000][d=1024][sigma=1][sparsity_fraction=0.5][mu_a=0][sigma_a=1]_sparse.meta
  dimension=1024
  binary=1

  run='LINEAR_REGRESSION'
  stepinitial_expbackoff=1
  stepinitial_decreasing=200

elif [ ${dataset,,} == higgs ]
then
  trainfile=./data/higgsq.bin
  # we don't care about the test sets nor metadata)
  testfile=./data/temp_meta.bin
  metafile=./data/temp.meta 
  dimension=140
  binary=1

  run='LOGISTIC_REGRESSION'
  #run='SVM'
  stepinitial_expbackoff=1
  stepinitial_decreasing=200

elif [ ${dataset,,} == rcv1 ]
then
  trainfile=./data/RCV1.train_shuffled.bin
  # we do care about the test sets but not about the metadata)
  testfile=./data/RCV1.test_shuffled.bin
  metafile=./data/temp.meta 
  dimension=47236
  binary=1

  run='LINEAR_REGRESSION'
  stepinitial_expbackoff=1
  stepinitial_decreasing=1000

elif [ ${dataset,,} == webspam_uni ]
then
  trainfile=./data/webspam_wc_normalized_unigram_shuffled.bin
  # we don't care about the test sets nor metadata)
  testfile=./data/temp_meta.bin
  metafile=./data/temp.meta 
  dimension=255
  binary=1

  run='LINEAR_REGRESSION'
  stepinitial_expbackoff=1
  stepinitial_decreasing=1000

elif [ ${dataset,,} == webspam_tri ]
then
  trainfile=./data/webspam_wc_normalized_trigram_shuffled.bin
  # we don't care about the test sets nor metadata)
  testfile=./data/temp_meta.bin
  metafile=./data/temp.meta 
  dimension=16609144
  binary=1

  run='LOGISTIC_REGRESSION'
  #run='LINEAR_REGRESSION'
  stepinitial_expbackoff=1
  stepinitial_decreasing=1000

else
  echo 'Dataset not specified correctly'
  exit
fi

if [ ${run^^} == LINEAR_REGRESSION ]
then
  execname="LINREG_"
elif [ ${run^^} == LOGISTIC_REGRESSION ]
then
  execname="LOGIT_"
elif [ ${run^^} == SVM ]
then
  execname="SVM_"
else
  echo 'RUN not specified correctly'
  exit
fi

if [ ${strategy^^} == PARAMETER_SERVER_SYNC ]
then
  execname=$execname"SYNC_"
elif [ ${strategy^^} == PARAMETER_SERVER_ASYNC ]
then
  execname=$execname"ASYNC_"
elif [ ${strategy^^} == PARAMETER_SERVER_ASYNC_SPARSE ]
then
  execname=$execname"ASYNCSPARSE_"
elif [ ${strategy^^} == ALL_REDUCE ]
then
  execname=$execname"ALLREDUCE_"
elif [ ${strategy^^} == ALL_REDUCE_SPARSE ]
then
  execname=$execname"ALLREDUCESPARSE_"
elif [ ${strategy^^} == HOGWILD ]
then
  execname=$execname"HOGWILD_"
else
  echo 'Strategy not specified correctly'
  exit
fi

if [ ${stepsize^^} == EXPBACKOFF ]
then
  execname=$execname"EXPBACKOFF_STEPSIZES_"
  stepinitial=$stepinitial_expbackoff
elif [ ${stepsize^^} == DECREASING ]
then
  execname=$execname"DECREASING_STEPSIZES_"
  stepinitial=$stepinitial_decreasing
else
  echo 'Stepsize behaviour not specified correctly'
  exit
fi

if [ ${representation^^} == DENSE ]
then
  execname=$execname"DENSE"
elif [ ${representation^^} == SPARSE ]
then
  execname=$execname"SPARSE"
else
  echo 'Representation not specified correctly'
  exit
fi

# run command!
set -x
$mpi_exec -n $nprocesses $execs_folder/$execname --binary $binary --dimension $dimension --beta $beta --epoch $nepochs --batch_size $batchsize --stepinitial $stepinitial --step_decay $stepdecay --lasso_regularizer $lasso_regularizer --quantization $quantization --qlevel $qlevel --splits $nthreads $trainfile $testfile $metafile
set +x

echo $execname
