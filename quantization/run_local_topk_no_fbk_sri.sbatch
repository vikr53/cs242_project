#!/bin/bash
#SBATCH -J topk_local_no_fbk
#SBATCH -o topk_local_no_fbk.out
#SBATCH -e topk_local_no_fbk.err
#SBATCH -p shared
#SBATCH -n 11
#SBATCH -t 1600
#SBATCH --mem-per-cpu=4000
srun -n $SLURM_NTASKS --mpi=pmix python3.7 local_top_k_no_fbk.py
