#!/bin/bash

echo "WARNING: If upgrading from v1.0 or v1.1 to v1.2. Note that chrombpnet has undergone linting to generate a modular structure for release on pypi.Hard-coded script paths are no longer necessary. Please refer to the updated README (below) to ensure your script calls are compatible with v1.2"

# exit when any command fails
set -e
set -o pipefail

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

reference_fasta=${1?param missing - reference_fasta}
bigwig_path=${2?param missing - bigwig_path}
overlap_peak=${3?param missing - overlap_peak}
nonpeaks=${4?param missing - nonpeaks}
fold=${5?param missing - fold}
bias_threshold_factor=${6?param missing - bias_threshold_factor}
output_dir=${7?param missing - output_dir}
filters=${8:-128}
n_dilation_layers=${9:-4}
seed=${10:-1234}
logfile=${11}

# defaults
inputlen=2114
outputlen=1000

function timestamp {
    # Function to get the current time with the new line character
    # removed 
    
    # current time
    date +"%Y-%m-%d_%H-%M-%S" | tr -d '\n'
}

# create the log file
if [ -z "$logfile" ]
  then
    echo "No logfile supplied - creating one"
    logfile=$output_dir"/train_bias_model.log"
    touch $logfile
fi

# this script does the following -  
# (1) filters your peaks/nonpeaks (removes outliers and removes edge cases and creates a new filtered set)
# (2) filters non peaks based on the given bias threshold factor
# (3) Calculates the counts loss weight 
# (4) Creates a TSV file that can be loaded into the next step
echo $( timestamp ): "chrombpnet_bias_hyperparams \\
       --genome=$reference_fasta \\
       --bigwig=$bigwig_path \\
       --peaks=$overlap_peak \\
       --nonpeaks=$nonpeaks \\
       --outlier_threshold=0.99 \\
       --chr_fold_path=$fold \\
       --inputlen=$inputlen \\
       --outputlen=$outputlen \\
       --max_jitter=0 \\
       --filters=$filters \\
       --n_dilation_layers=$n_dilation_layers \\
       --bias_threshold_factor=$bias_threshold_factor \\
       --output_dir $output_dir" | tee -a $logfile

chrombpnet_bias_hyperparams \
    --genome=$reference_fasta \
    --bigwig=$bigwig_path \
    --peaks=$overlap_peak \
    --nonpeaks=$nonpeaks \
    --outlier_threshold=0.99 \
    --chr_fold_path=$fold \
    --inputlen=$inputlen \
    --outputlen=$outputlen \
    --max_jitter=0 \
    --filters=$filters \
    --n_dilation_layers=$n_dilation_layers \
    --bias_threshold_factor=$bias_threshold_factor \
    --output_dir $output_dir | tee -a $logfile