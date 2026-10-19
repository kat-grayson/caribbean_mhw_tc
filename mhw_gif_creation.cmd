#!/bin/bash
#SBATCH --account=bk1580
#SBATCH --time=08:00:00
#SBATCH --mem=128GB
#SBATCH --partition=shared
#SBATCH --job-name=gif
#SBATCH --output=gif.out
#SBATCH --error=gif.err

set -e 

# load modules
eval "$(conda shell.bash hook)"
# conda activate aqua

# conda init bash
# conda deactivate
# source /home/b/b382291/aqua_26/bin/activate

# run python script
python -u /home/b/b382291/git/small_islands/mhw_gif_creation.py

