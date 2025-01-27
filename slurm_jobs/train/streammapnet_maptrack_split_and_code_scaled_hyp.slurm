#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=305:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu80g

#SBATCH --mem=50G              # 5 gigabytes memory
#SBATCH --output=/cluster/home/terjenf/StreamMapNet/slurm_outputs/streammapnet_maptrack_split_scaled_hyp.out    # Log file

#SBATCH --job-name="SteamMapNet Maptrack Split Scaled hyp"
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate streammapnet

python -m torch.distributed.launch --nproc_per_node 1 --master_port 29520 /cluster/home/terjenf/StreamMapNet/tools/train.py /cluster/home/terjenf/StreamMapNet/plugin/configs/nusc_newsplit_480_60x30_24e_maptrack_split_scaled_hyp.py --launcher pytorch --gpus 1 --work-dir /cluster/home/terjenf/StreamMapNet/final_work