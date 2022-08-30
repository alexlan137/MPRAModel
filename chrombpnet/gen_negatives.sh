#!/bin/bash
#$ -S /bin/bash
#$ -o /wynton/home/corces/allan/MPRAModel/chrombpnet/wynton_logs
#$ -cwd
#$ -j y
#$ -l mem_free=50G
#$ -l h_rt=24:00:00

step3_get_background_regions.sh data/reference/hg38.genome.fa data/reference/hg38.chrom.sizes data/exclusion.bed data/peaks.bed 2114 data/utils/genome2114.bed data/negatives_data data/splits/fold_0.json