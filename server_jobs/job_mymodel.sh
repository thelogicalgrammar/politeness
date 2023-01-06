#!/bin/bash
#SBATCH --partition=single
#SBATCH --ntasks=4
#SBATCH --time=10:00:00

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
python -u my_model_functions.py