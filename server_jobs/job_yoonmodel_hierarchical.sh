#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=4
#SBATCH --time=50:00:00

module load devel/miniconda/3
source $MINICONDA_HOME/etc/profile.d/conda.sh

conda deactivate 
conda activate pymc_env

echo $(conda env list)
echo " "
echo $(module list)
echo " "
echo $(which python)
echo " "

cd ../
python -u yoon_model_functions_hierarchical.py