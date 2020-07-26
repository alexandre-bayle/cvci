#!/bin/bash

sample_sizes=(700 1000 1500 2300 3400 5000 7500 11000)

task=$1
algo=$2 # can be either single algo or algo comp
n_reps=$3
path_to_res=$4

path=$path_to_res"/"$task"/"$algo

for n in ${sample_sizes[@]}
do
    path2=$path"/n"$n
    if [[ ! -f $path2"/all_reps.h5" ]] || [[ ! -f $path2"/cond_indiv_err.h5"  ]]
    then
	echo -e "Combining for n="$n" in this run"
	command="sbatch --mem=4G -c 1 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J COMB"$task"a"$algo"n"$n" -o "$path2"/combine.out --time=01:30:00 combineOneExper.sh $task $algo $n $n_reps $path_to_res"
	$command
    else
        echo -e "Combining for n="$n" already done in a previous run"
    fi
done
