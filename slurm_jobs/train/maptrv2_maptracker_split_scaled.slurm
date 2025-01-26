#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=305:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu80g

#SBATCH --mem=200G              # 5 gigabytes memory
#SBATCH --output=/cluster/home/terjenf/maptr_new/slurm_outputs/maptrv2_maptracker_split_scaled_hyp.out    # Log file

#SBATCH --job-name="Maptrv2 Maptracker Split"
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate maptr_new

python /cluster/home/terjenf/maptr_new/tools/train.py /cluster/home/terjenf/maptr_new/projects/configs/maptrv2/maptrv2_nusc_r50_1ep_maptracker_scaled_hyp.py --work-dir /cluster/home/terjenf/maptracker/master_work