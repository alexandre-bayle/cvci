#!/bin/bash

algos_Clf=("LR" "RF" "NN")
algos_Reg=("RF" "NN" "RR")

task=$1
n_reps=$2
path_to_res=$3

path=$path_to_res"/"$task

if [[ $task = "Clf" ]]
then
    algos=${algos_Clf[@]}
else
    algos=${algos_Reg[@]}
fi

for algo in ${algos[@]}
do
    echo -e $algo
    command="sbatch --mem=4G -c 1 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J COMB"$task"a"$algo" -o "$path"/"$algo"/combine.out --time=00:10:00 combineAllIntermExper.sh $task $algo $n_reps $path_to_res"
    $command
    for algo2 in ${algos[@]}
    do
	if [[ $algo != $algo2 ]]
	then
	    comp=$algo"_"$algo2
	    echo -e $comp
	    command="sbatch --mem=4G -c 1 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J COMB"$task"c"$comp" -o "$path"/"$comp"/combine.out --time=00:10:00 combineAllIntermExper.sh $task $comp $n_reps $path_to_res"
	    $command
	fi
    done
done
