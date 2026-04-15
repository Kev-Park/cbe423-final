#!/bin/bash
#SBATCH --job-name=pred_train    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G                # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of GPUs per node
#SBATCH --constraint=a100
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kp0374@princeton.edu

module purge

source .venv/bin/activate

python train_predictor.py --epochs=100 --batch_size=128 --lr=1e-5