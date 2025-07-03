#!/bin/bash

#SBATCH --job-name=dinov2_autoencoder # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # total number of tasks per node
#SBATCH --cpus-per-task=1 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node (matching --num_processes in accelerate launch)
#SBATCH --time=60:00:00 # total run time limit (HH:MM:SS)
#SBATCH --output=dinov2_autoencoder.out
#SBATCH --error=dinov2_autoencoder.err
#SBATCH --mail-type=begin # send email when job begins
#SBATCH --mail-type=end # send email when job ends
#SBATCH --mail-user=sunny17@kaist.ac.kr

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Uncomment and modify the following line if you need to activate a conda environment
# conda activate torch-env

srun accelerate launch --multi_gpu --mixed_precision=fp16 --num_machines=1 --num_processes=8 --machine_rank=0 --main_process_port=29600 train.py