#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=305:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu40g

#SBATCH --mem=50G              # 5 gigabytes memory
#SBATCH --output=/cluster/home/terjenf/StreamMapNet/slurm_outputs/streammapnet_maptrv2_py.out    # Log file

#SBATCH --job-name="SteamMapNet Maptrv2 py"
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate streammapnet

python /cluster/home/terjenf/StreamMapNet/tools/train.py /cluster/home/terjenf/StreamMapNet/plugin/configs/nusc_newsplit_480_60x30_24e_maptrv2.py --work-dir /cluster/home/terjenf/StreamMapNet/new_master_work