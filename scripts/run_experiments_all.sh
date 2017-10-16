nr=$1
output_base_folder=results
prevent_rerun=true
remake=false

if [ -z $nr ]
then
  echo "Experiment nr. not specified!!"
  exit
fi

folder=$output_base_folder/$nr

if $prevent_rerun && [ -d $folder ]
then
  echo "Experiment nr. has already been used!!"
  exit
fi

if [ -d $folder ]
then
  rm -r -f $folder
fi

mkdir -p $folder

start=$(date)
startsec=$(date +%s)

git_info=$(git show -s --format=medium)
git_branch=$(git rev-parse --abbrev-ref HEAD)

info_log="$folder/experiment_info.log"
echo "START: $start" > $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo >> $info_log 2>&1
echo "GIT INFOS (Branch: $git_branch)" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo "$git_info" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1

if $remake
then
  echo >> $info_log 2>&1
  echo "MAKE CLEAN" >> $info_log 2>&1
  echo "-------------------------------" >> $info_log 2>&1
  make clean >> $info_log 2>&1
  echo "-------------------------------" >> $info_log 2>&1
  echo "MAKE ALL" >> $info_log 2>&1
  make >> $info_log 2>&1
  echo "-------------------------------" >> $info_log 2>&1
fi
echo >> $info_log 2>&1

# RUN ALL SUB_EXPERIMENTS
nprocessesList=`seq 1 80`
nthreadsList=(1)
takeList=`seq 1 10`

batchsize=10
nepochs=20

###############################
########## Synthetic ##########
###############################

########## PARAMETER_SERVER_SYNC SYNTHETIC EXPBACKOFF_STEPSIZE ##########
# echo "PARAMETER_SERVER_SYNC SYNTHETIC EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# for nprocesses in "${nprocessesList[@]}"; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/1_parameter_server_sync/1_exp_backoff_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic parameter_server_sync expbackoff dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1
# 
# ########## PARAMTER_SERVER_SYNC SYNTHETIC DECREASING_STEPSIZE ##########
# echo "PARAMETER_SERVER_SYNC SYNTHETIC DECREASING_STEPSIZE" >> $info_log 2>&1
# for nprocesses in "${nprocessesList[@]}"; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/1_parameter_server_sync/2_decreasing_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic parameter_server_sync decreasing dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

# ########## PARAMTER_SERVER_ASYNC SYNTHETIC EXPBACKOFF_STEPSIZE ##########
# echo "PARAMETER_SERVER_ASYNC SYNTHETIC EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# #for nprocesses in "${nprocessesList[@]}"; do
# for nprocesses in $nprocessesList; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/2_parameter_server_async/1_exp_backoff_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic parameter_server_async expbackoff dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

# ########## PARAMTER_SERVER_ASYNC SYNTHETIC DECREASING_STEPSIZE ##########
# echo "PARAMETER_SERVER_ASYNC SYNTHETIC DECREASING_STEPSIZE" >> $info_log 2>&1
# for nprocesses in "${nprocessesList[@]}"; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/2_parameter_server_async/2_decreasing_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic parameter_server_async decreasing dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

# ########## ALL_REDUCE SYNTHETIC EXPBACKOFF_STEPSIZE ##########
# echo "ALL_REDUCE SYNTHETIC EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# #for nprocesses in "${nprocessesList[@]}"; do
# for nprocesses in $nprocessesList; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/3_all_reduce/1_exp_backoff_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic all_reduce expbackoff dense $nprocesses $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

# ########## ALL_REDUCE SYNTHETIC DECREASING_STEPSIZE ##########
# echo "ALL_REDUCE SYNTHETIC DECREASING_STEPSIZE" >> $info_log 2>&1
# for nprocesses in "${nprocessesList[@]}"; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/1_synthetic/3_all_reduce/2_decreasing_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh synthetic all_reduce decreasing dense $nprocesses $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

#####################################
########## HIGGS - DATASET ##########
#####################################

# ########## PARAMTER_SERVER_ASYNC HIGGS EXPBACKOFF_STEPSIZE ##########
# echo "PARAMETER_SERVER_ASYNC HIGGS EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# for nprocesses in $nprocessesList; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/2_higgs/2_parameter_server_async/1_exp_backoff_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh higgs parameter_server_async expbackoff dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1
# 
# ########## ALL_REDUCE HIGGS EXPBACKOFF_STEPSIZE ##########
# echo "ALL_REDUCE HIGGS EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# #for nprocesses in "${nprocessesList[@]}"; do
# for nprocesses in $nprocessesList; do
#   for nthreads in ${nthreadsList[@]}; do
#     base_path="$folder/2_higgs/3_all_reduce/1_exp_backoff_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
#     mkdir -p $base_path
#     command="bash scripts/run_single_experiment.sh higgs all_reduce expbackoff dense $nprocesses $nthreads $nepochs $batchsize"
#     printf "\t$command - " >> $info_log 2>&1
#     for take in $takeList; do
#       output_name="$base_path/[take=$take]"
#       $command > $output_name.log 2>&1
#       printf "*" >> $info_log 2>&1
#     done
#     echo >> $info_log 2>&1
#   done
# done
# echo "-------------------------------" >> $info_log 2>&1

########## PARAMTER_SERVER_ASYNC HIGGS DECREASING_STEPSIZE ##########
echo "PARAMETER_SERVER_ASYNC HIGGS DECREASING_STEPSIZE " >> $info_log 2>&1
for nprocesses in $nprocessesList; do
  for nthreads in ${nthreadsList[@]}; do
    base_path="$folder/2_higgs/2_parameter_server_async/2_decreasing_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
    mkdir -p $base_path
    command="bash scripts/run_single_experiment.sh higgs parameter_server_async decreasing dense $(($nprocesses+1)) $nthreads $nepochs $batchsize"
    printf "\t$command - " >> $info_log 2>&1
    for take in $takeList; do
      output_name="$base_path/[take=$take]"
      $command > $output_name.log 2>&1
      printf "*" >> $info_log 2>&1
    done
    echo >> $info_log 2>&1
  done
done
echo "-------------------------------" >> $info_log 2>&1

########## ALL_REDUCE HIGGS DECREASING_STEPSIZE ##########
echo "ALL_REDUCE HIGGS DECREASING_STEPSIZE " >> $info_log 2>&1
#for nprocesses in "${nprocessesList[@]}"; do
for nprocesses in $nprocessesList; do
  for nthreads in ${nthreadsList[@]}; do
    base_path="$folder/2_higgs/3_all_reduce/2_decreasing_stepsizes/linreg[p=$nprocesses][t=$nthreads]"
    mkdir -p $base_path
    command="bash scripts/run_single_experiment.sh higgs all_reduce decreasing dense $nprocesses $nthreads $nepochs $batchsize"
    printf "\t$command - " >> $info_log 2>&1
    for take in $takeList; do
      output_name="$base_path/[take=$take]"
      $command > $output_name.log 2>&1
      printf "*" >> $info_log 2>&1
    done
    echo >> $info_log 2>&1
  done
done
echo "-------------------------------" >> $info_log 2>&1

# ########## ALL_REDUCE HIGGS EXPBACKOFF_STEPSIZE CHANGE OF BATCH_SIZE  ##########
# echo "ALL_REDUCE HIGGS EXPBACKOFF_STEPSIZE" >> $info_log 2>&1
# batchizeList=`seq 10 10 800`
# for batchsize in $batchizeList; do
#   base_path="$folder/2_higgs/3_all_reduce/1_exp_backoff_stepsizes/linreg[b=$batchsize]"
#   mkdir -p $base_path
#   command="bash scripts/run_single_experiment.sh higgs all_reduce expbackoff dense 1 1 $nepochs $batchsize"
#   printf "\t$command - " >> $info_log 2>&1
#   for take in $takeList; do
#     output_name="$base_path/[take=$take]"
#     $command > $output_name.log 2>&1
#     printf "*" >> $info_log 2>&1
#   done
#   echo >> $info_log 2>&1
# done
# echo "-------------------------------" >> $info_log 2>&1

# OUPUT DURATION
end=$(date)
endsec=$(date +%s)
diff=$(( endsec - startsec ))

echo >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo "END: $end (DURATION: $diff seconds)" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
