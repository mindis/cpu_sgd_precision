# SETUP EXPERIMENT
nr=$1
output_base_folder=results
prevent_rerun=false
remake=false

if [ -z $nr ]
then
  echo "Experiment nr. as first argument not specified!!"
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
info_log="$folder/experiment_info.log"

start=$(date)
startsec=$(date +%s)

bash scripts/setup_experiment.sh $info_log "$start" $remake

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

nprocessesList=(1 3)
nthreadsList=(1 2)
nepochs=20
batchsize=10

DATASET=("synthetic" "higgs")
STRATEGYPM=("parameter_server_sync" "parameter_server_async")
STEPSIZE=("decreasing" "expbackoff")
REP=("dense" "sparse")

for dat in "${DATASET[@]}"; do
  for str in "${STRATEGYPM[@]}"; do
    for ste in "${STEPSIZE[@]}"; do
      for rep in "${REP[@]}"; do
        for nprocesses in "${nprocessesList[@]}"; do
          for nthreads in "${nthreadsList[@]}"; do
            command="bash scripts/run_single_experiment.sh $dat $str $ste $rep $((nprocesses+1)) $nthreads $nepochs $batchsize"
            echo $command >> $info_log 2>&1
            output_name="$folder/[$dat][$str][$ste][$rep][p=$nprocesses][t=$nthreads]"
            $command > $output_name.log 2>&1
          done
        done
      done
    done
  done
  for ste in "${STEPSIZE[@]}"; do
    for rep in "${REP[@]}"; do
      for nprocesses in "${nprocessesList[@]}"; do
        for nthreads in "${nthreadsList[@]}"; do
          command="bash scripts/run_single_experiment.sh $dat all_reduce $ste $rep $nprocesses $nthreads $nepochs $batchsize"
          echo $command >> $info_log 2>&1
          output_name="$folder/[$dat][all_reduce][$ste][$rep][p=$nprocesses][t=$nthreads]"
          $command > $output_name.log 2>&1
        done
      done
    done
  done
done

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# END EXPERIMENT
bash scripts/finish_experiment.sh $info_log $startsec
