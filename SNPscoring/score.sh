#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/wynton
#$ -cwd
#$ -j y
#$ -l mem_free=5G
#$ -l h_rt=3:00:00

python3 CBPscoring.py