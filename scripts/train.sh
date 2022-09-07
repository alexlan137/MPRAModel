#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/wynton
#$ -cwd
#$ -j y
#$ -l mem_free=50G
#$ -l h_rt=24:00:00

LD_LIBRARY_PATH=/wynton/home/corces/allan/.conda/envs/MPRAModel/lib
python3 utils/train_mpra.py