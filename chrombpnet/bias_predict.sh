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
logfile=/wynton/home/corces/allan/MPRAModel/chrombpnet/models/bias_model/trainbias.log

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


# predictions and metrics on the bias model trained
echo $( timestamp ): "chrombpnet_predict \\
        --genome=$reference_fasta \\	 
        --bigwig=$bigwig_path \\  
        --nonpeaks=$output_dir/filtered.bias_nonpeaks.bed \\
        --chr_fold_path=$fold \\
        --inputlen=$inputlen \\
        --outputlen=$outputlen \\
        --output_prefix=$output_dir/bias \\
        --batch_size=256 \\
        --model_h5=$output_dir/bias.h5" | tee -a $logfile

chrombpnet_predict \
    --genome=$reference_fasta \
    --bigwig=$bigwig_path \
    --nonpeaks=$output_dir/filtered.bias_nonpeaks.bed \
    --chr_fold_path=$fold \
    --inputlen=$inputlen \
    --outputlen=$outputlen \
    --output_prefix=$output_dir/bias \
    --batch_size=256 \
    --model_h5=$output_dir/bias.h5 | tee -a $logfile