#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/chrombpnet/wynton_logs
#$ -cwd
#$ -j y
#$ -l mem_free=50G
#$ -l h_rt=24:00:00

step6_train_chrombpnet_model.sh data/reference/hg38.genome.fa data/GM12878v2.bw data/peaks.bed data/negatives_data/negatives_with_summit.bed data/splits/fold_0.json models/bias_model/bias.h5 models/chrombpnet_model2 ATAC_PE