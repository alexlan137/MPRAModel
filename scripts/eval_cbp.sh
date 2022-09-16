#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/wynton
#$ -cwd
#$ -j y
#$ -l mem_free=10G
#$ -l h_rt=2:00:00

python3 utils/eval_cbp.py
python3 utils/eval_cbpv1.py