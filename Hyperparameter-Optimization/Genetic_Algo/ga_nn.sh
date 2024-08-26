#!/bin/sh
module load anaconda3/5.3.1
export PYTHONUNBUFFERED=x
DATASET=$1 
/home/../.../.conda/envs/datascience/bin/python  path/ga_nn.py --dataset "${DATASET}" --data_path "../data/"
echo "job started"