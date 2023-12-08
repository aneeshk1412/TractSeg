#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   * Serial Job on SKX Normal Queue *
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J tractseg_boundary_dou_train                        # Job name
#SBATCH -o tractseg_boundary_dou_train.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e tractseg_boundary_dou_train.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100                   # Queue (partition) name
#SBATCH -N 1                            # Total # of nodes (must be 1 for serial)
#SBATCH -n 4				# Number of cores
#SBATCH -t 24:00:00                     # Run time (hh:mm:ss)
#SBATCH --mail-user=aneeshks@utexas.edu
#SBATCH --mail-type=all                 # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A IRI23014         # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module load python3

cd $WORK/TractSeg/
source .venv/bin/activate
pip install .

# Launch serial code...

cd $SCRATCH
ExpRunner --config my_custom_experiment

# ---------------------------------------------------
# sbatch -J tractseg_train -o tractseg_train.o%j -e tractseg_train.e%j -p gpu-a100 -N 1 -n 32 -t 24:00:00 --mail-user=aneeshks@utexas.edu --mail-type=all -A IRI23014 ../SLURM_FILES/training.slurm
