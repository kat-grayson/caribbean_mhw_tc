#!/bin/bash
#SBATCH --account=bk1580
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --partition=shared
#SBATCH --job-name=mwh
#SBATCH --output=mhw.out
#SBATCH --error=mhw.err

set -e 

# load modules
eval "$(conda shell.bash hook)"
# conda activate aqua

# conda init bash
# conda deactivate
# source /home/b/b382291/aqua_26/bin/activate

# run python script
python -u /home/b/b382291/git/small_islands/run_3d_mhw.py

