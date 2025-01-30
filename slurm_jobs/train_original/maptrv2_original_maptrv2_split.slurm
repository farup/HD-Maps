#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=305:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1 
#SBATCH --constraint=gpu80g

#SBATCH --mem=50G              
#SBATCH --output=/cluster/home/terjenf/maptr_new/slurm_outputs/maptrv2_original_maptrv2_split_24ep.out    # Log file

#SBATCH --job-name="Maptrv2 Original Maptrv2-Split"
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate maptr_new

python /cluster/home/terjenf/maptr_new/tools/train.py /cluster/home/terjenf/maptr_new/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_maptrv2.py --work-dir /cluster/home/terjenf/maptr_new/final_Work