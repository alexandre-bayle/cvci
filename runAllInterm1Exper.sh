#!/bin/bash

task=$1
n=$2
k=$3
n_reps=$4
path_to_bash_outputs=$5
path_to_res=$6
path_to_data=$7
LOOCV=$8

n_rep_chunks=20 # used to avoid submitting jobs for all replications sequentially and instead do it by consecutive chunks sent to other cores, so choose n_rep_chunks such that it evenly divides the number of replications

sub_size=$((n_reps/n_rep_chunks)) # ideally you want sub_size to be approximately equal to sqrt(n_reps) so choose n_rep_chunks wisely

for i in $(seq 1 $n_rep_chunks)
do
    start=$(((i-1)*sub_size+1))
    end=$((i*sub_size))
    command="sbatch --mem=4G -c 1 -p shared,janson,janson_cascade,janson_bigmem,serial_requeue -J CV"$task"n"$n"k"$k"ch"$i" -o "$path_to_bash_outputs"/chunk"$i".out -e "$path_to_bash_outputs"/chunk"$i".err --mail-type=FAIL --time=00:05:00 runAllInterm2Exper.sh $task $n $k $n_reps $start $end $path_to_bash_outputs $path_to_res $path_to_data $LOOCV"
    $command
    sleep .5
done
