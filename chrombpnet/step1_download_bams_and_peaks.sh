#!/bin/bash

echo "WARNING: If upgrading from v1.0 or v1.1 to v1.2. Note that chrombpnet has undergone linting to generate a modular structure for release on pypi.Hard-coded script paths are no longer necessary. Please refer to the updated README (below) to ensure your script calls are compatible with v1.2"


# exit when any command fails
set -e
set -o pipefail
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT
data_dir=${1?param missing - data_dir}

# download bam
wget https://www.encodeproject.org/files/ENCFF077FBI/@@download/ENCFF077FBI.bam -O $data_dir/rep1.bam
wget https://www.encodeproject.org/files/ENCFF128WZG/@@download/ENCFF128WZG.bam -O $data_dir/rep2.bam
wget https://www.encodeproject.org/files/ENCFF534DCE/@@download/ENCFF534DCE.bam -O $data_dir/rep3.bam

samtools merge -f $data_dir/merged_unsorted.bam $data_dir/rep1.bam $data_dir/rep2.bam $data_dir/rep3.bam
samtools sort -@4 $data_dir/merged_unsorted.bam -o $data_dir/merged.bam
samtools index $data_dir/merged.bam

# download overlap peaks (default peaks on ENCODE)
wget https://www.encodeproject.org/files/ENCFF333TAT/@@download/ENCFF333TAT.bed.gz -O $data_dir/overlap.bed.gz

# download reference data
wget https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz -O $data_dir/hg38.fa.gz
yes n | gunzip $data_dir/hg38.fa.gz

# download reference chromsome sizes
wget https://www.encodeproject.org/files/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv -O $data_dir/hg38.chrom.sizes

# download blacklist regions 
wget https://www.encodeproject.org/files/ENCFF356LFX/@@download/ENCFF356LFX.bed.gz -O $data_dir/blacklist.bed.gz
