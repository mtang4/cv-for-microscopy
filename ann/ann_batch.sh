#!/bin/sh
#SBATCH -J matjob

#bash script assumes the code was compiled with MATLAB 2020b
export MALLOC_CHECK = 0
tangm6/data/ann/run_ann_approach_final.sh /usr/local/matlab-compiler/v99
