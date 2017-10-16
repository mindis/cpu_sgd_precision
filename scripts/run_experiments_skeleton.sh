# SETUP EXPERIMENT
nr=$1
output_base_folder=results
prevent_rerun=true
remake=true

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
# PUT YOUR STUFF HERE
# --------------------------------



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# END EXPERIMENT
bash scripts/finish_experiment.sh $info_log $startsec
