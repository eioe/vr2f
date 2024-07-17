#!/bin/bash -l 
# specify the indexes of the job array elements
#SBATCH --array=0-35
# Standard output and error: 
#SBATCH -o ./job.out.%j        # Standard output, %A = job ID, %a = job array index 
#SBATCH -e ./job.err.%j        # Standard error, %A = job ID, %a = job array index 
# Initial working directory: 
#SBATCH -D /ptmp/fklotzsche/Experiments/vrstereofem/
# Job Name: 
#SBATCH -J hansi_ftw
# Queue (Partition): 
#SBATCH --partition=general 
# Number of nodes and MPI tasks per node: 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=20000MB
# 
#SBATCH --mail-type=all 
#SBATCH --mail-user=klotzsche@cbs.mpg.de 
# 
#SBATCH --time=02:00:00
 
module load anaconda/3/2023.03
module load mkl
conda activate vr2fem
 
# Run the program: 
srun python3.10 ./vr2f/code/vr2f/decoding/decoding_calc_source_timecourses.py $SLURM_ARRAY_TASK_ID