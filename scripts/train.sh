#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/wynton
#$ -cwd
#$ -j y
#$ -l mem_free=10G
#$ -l h_rt=0:15:00

LD_LIBRARY_PATH=/wynton/home/corces/allan/.conda/envs/MPRAModel/lib
python3 MPRA_model_development/run_model.py