#!/usr/bin/bash
source activate py3.6_torch1.0_pyro0.3

opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

outdir="./cifar10-cnn-svgd-checkpoint"

echo "Output directory: $outdir"

script="python -u cnn_svgd_test.py"

outs="--output=$outdir/cnn_svgd.out --error=$outdir/cnn_svgd.err"

sbatch $opts $outs --wrap="python -u cnn_svgd_test.py"


