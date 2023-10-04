#!/bin/sh
#
#SBATCH --job-name="pmf_job_1"
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-3me-msc-ro
#SBATCH --output=hello_world_mpi.%j.out
#SBATCH --error=hello_world_mpi.%j.err

module load miniconda3/4.12.0
conda activate pmf

srun python --version
srun conda list