#!/bin/bash
#SBATCH --account=bk1580
#SBATCH --time=08:00:00
#SBATCH --mem=200GB
#SBATCH --partition=shared
#SBATCH --job-name=mwh_hr
#SBATCH --output=mhw_hr.out
#SBATCH --error=mhw_hr.err

set -e 

# load modules
eval "$(conda shell.bash hook)"
# conda activate aqua

# conda init bash
# conda deactivate
# source /home/b/b382291/aqua_26/bin/activate

# run python script
python -u /home/b/b382291/git/small_islands/run_3d_mhw_highres.py

