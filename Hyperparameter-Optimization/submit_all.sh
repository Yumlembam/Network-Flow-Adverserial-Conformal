#!/bin/bash
# First dataset
sbatch --partition=120hour --cpus-per-task=28 --mem-per-cpu=64000MB --ntasks-per-node=1 --nodes=1 --mem=64000MB --output=ga_iscx_nn.out ga_nn.sh "iscx"
# Third dataset
sbatch --partition=120hour --cpus-per-task=28 --mem-per-cpu=64000MB --ntasks-per-node=1 --nodes=1 --mem=64000MB --output=ga_isot_nn.out ga_nn.sh "isot_botnet"