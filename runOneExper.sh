#!/bin/bash

task=$1
n=$2
k=$3
rep=$4
path_to_res=$5
path_to_data=$6
LOOCV=$7

module load <my_Anaconda_installation> # add your Anaconda installation
source activate <my_environment> # add your environment

echo -e "Task is "$task", sample size is "$n", number of folds is "$k", index of the replication is  "$rep"...\n"

python cv_num_exper.py $task $n $k $rep $path_to_res $path_to_data $LOOCV
