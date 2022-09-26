#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/chrombpnet/wynton_logs
#$ -cwd
#$ -j y
#$ -l mem_free=50G
#$ -l h_rt=24:00:00

bias_train.sh data/reference/hg38.genome.fa data/Cluster24/Cluster24.bpnet.unstranded.bw data/Cluster24/Cluster24.idr.optimal.narrowPeak data/Cluster24/negatives_data/negatives_with_summit.bed data/splits/fold_0.json 0.5 models/cluster24biasmodel
bias_predict.sh data/reference/hg38.genome.fa data/Cluster24/Cluster24.bpnet.unstranded.bw data/Cluster24/Cluster24.idr.optimal.narrowPeak data/Cluster24/negatives_data/negatives_with_summit.bed data/splits/fold_0.json 0.5 models/cluster24biasmodel