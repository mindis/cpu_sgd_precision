info_log=$1
start=$2

git_info=$(git show -s --format=medium)
git_branch=$(git rev-parse --abbrev-ref HEAD)

echo "START: $start" > $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo >> $info_log 2>&1
echo "GIT INFOS (Branch: $git_branch)" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1
echo "$git_info" >> $info_log 2>&1
echo "-------------------------------" >> $info_log 2>&1

if $3
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
