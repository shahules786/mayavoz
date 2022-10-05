#!/bin/bash
set -e

echo "Loading Anaconda Module"
module load anaconda

echo "Creating Virtual Environment"
conda env create -f environment.yml ||  conda env update -f environment.yml

source activate enhancer

echo "copying files"
# cp /scratch/$USER/TIMIT/.* /deep-transcriber
