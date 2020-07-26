#!/bin/bash

sample_sizes=(11000 7500 5000 3400 2300 1500 1000 700)

task=$1
n_reps=$2
path_to_res=$3
path_to_data=$4
LOOCV=$5

path=$path_to_res"/"
[ ! -d $path ] && mkdir $path
path+=$task"/"
[ ! -d $path ] && mkdir $path
path+="outputs/"
[ ! -d $path ] && mkdir $path

for n in ${sample_sizes[@]}
do
    k=10
    echo -e "Sample size: "$n", number of folds: "$k
    path_to_bash_outputs=$path"/n"$n"/"
    [ ! -d $path_to_bash_outputs ] && mkdir $path_to_bash_outputs
    command="sbatch --mem=4G -c 1 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J CV"$task"n"$n"k"$k" -o "$path_to_bash_outputs"/all.out -e "$path_to_bash_outputs"/all.err --time=00:05:00 runAllInterm1Exper.sh $task $n $k $n_reps $path_to_bash_outputs $path_to_res $path_to_data $LOOCV"
    $command
    sleep .5
done
