#!/bin/sh
#
#SBATCH --job-name="pmf_job_1"
#SBATCH --account=education-3me-msc-ro
#SBATCH --partition=gpu
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=1G

module load miniconda3/4.12.0
conda activate pmf

srun python --version
srun python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --output outputs/5way-5shot-10-query --dataset custom --epoch 100 --arch dino_small_patch16 --device cuda --fp16 --nClsEpisode 5 --nSupport 5 --nQuery 10 --nEpisode 2000 --sched cosine --lr 5e-5 --warmup-lr 1e-6 --min-lr 1e-6 --warmup-epochs 5 --num_workers 10