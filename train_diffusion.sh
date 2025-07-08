#!/bin/bash

#SBATCH --job-name="optflow-diffusion"
#SBATCH --gres=shard:2 # 6 shards = 1 GPU
#SBATCH --cpus-per-task=2 # ML1 has ? CPUs
#SBATCH --mem=10GB # ML2 has RealMemory=512000 (MB)
#SBATCH --oversubscribe
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aamal.hussain@physicsx.ai
#SBATCH -o /home/aamal.hussain/flow_research/outputs/logs/slurm-%j.out


PROJECT_DIR=/home/aamal.hussain/flow_research
CONDA_ENV_NAME=optflow

cd $PROJECT_DIR
echo Current directory $(pwd)
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
source /home/aamal.hussain/.bashrc
conda activate $CONDA_ENV_NAME

python -m train.diffusion \
            dataset=dora  \
            batch_size=4
