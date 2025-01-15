#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=50:15:00     
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=2     # 2 CPU cores
#SBATCH --mem=50G              # 5 gigabytes memory
#SBATCH --output=/cluster/home/terjenf/maptracker/slurm_outputs/annotations/hello.txt    # Log file

#SBATCH --job-name="NuScence Annotaions"
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate maptracker

python /cluster/home/terjenf/maptracker/tools/data_converter/nuscenes_converter.py --data-root /cluster/home/terjenf/maptracker/data/nuscenes