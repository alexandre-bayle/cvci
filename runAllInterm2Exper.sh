#!/bin/bash

task=$1
n=$2
k=$3
n_reps=$4
start=$5
end=$6
path_to_bash_outputs=$7
path_to_res=$8
path_to_data=$9
LOOCV=${10}

for rep in $(seq $start $end)
do
    if [[ ! -f $path_to_res"/"$task"/RF_NN/n"$n"/rep"$rep".h5" ]] || [[ ! -f $path_to_res"/"$task"/RF_NN/n"$n"/rep"$rep"_sigma.h5" ]] # in order to not rerun jobs that already worked in previous runs (the choice RF_NN is due to the fact that it is used in both classification and regression)
    then
	echo -e "Running replication "$rep" in this run"
	command="sbatch --mem=8G -c 2 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J CV"$task"n"$n"k"$k"rep"$rep" -o "$path_to_bash_outputs"/rep"$rep".out -e "$path_to_bash_outputs"/rep"$rep".err --mail-type=FAIL --time=01:30:00 runOneExper.sh $task $n $k $rep $path_to_res $path_to_data $LOOCV"
	$command
	sleep .5
    else
	echo -e "Replication "$rep" already worked in a previous run"
    fi
done
