#!/bin/bash

task=$1
algo=$2
n=$3
n_reps=$4
path_to_res=$5

module load <my_Anaconda_installation> # add your Anaconda installation
source activate <my_environment> # add your environment

echo -e "Combining "$n_reps" replications for algo/comp = "$algo", n = "$n", task = "$task"...\n"

python combine_results.py $task $algo $n $n_reps $path_to_res
