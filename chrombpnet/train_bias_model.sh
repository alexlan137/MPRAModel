#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/chrombpnet/wynton_logs
#$ -cwd
#$ -j y
#$ -l mem_free=50G
#$ -l h_rt=24:00:00

bias_train.sh data/reference/hg38.genome.fa data/GM12878v2.bw data/peaks.bed data/negatives_data/negatives_with_summit.bed data/splits/fold_0.json 0.5 models/bias_model
bias_predict.sh data/reference/hg38.genome.fa data/GM12878v2.bw data/peaks.bed data/negatives_data/negatives_with_summit.bed data/splits/fold_0.json 0.5 models/bias_model