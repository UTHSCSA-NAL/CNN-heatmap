#!/bin/bash

#-------------------------------------------------------------------------------
# hello
#-------------------------------------------------------------------------------

#SBATCH -J test                      # Job name
#SBATCH -o log/%xout.txt	          # Name of stdout output file
#SBATCH -e log/%xerr.txt	          # Name of stderr error file
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 99:99:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=wangd2@livemail.uthscsa.edu   # Email
#SBATCH --mail-type=all                   # Send email at begin and end of job



module load cuda10.0/blas/10.0.130                
module load cuda10.0/fft/10.0.130                 
module load cuda10.0/nsight/10.0.130              
module load cuda10.0/profiler/10.0.130            
module load cuda10.0/toolkit/10.0.130 
module load python36

source /home/wangd2/envs/torch/bin/activate
#pip install numba



pwd
date
which python3

python3 test.py [0,1,2]



