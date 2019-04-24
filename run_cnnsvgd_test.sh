#!/usr/bin/bash
module load python/3.5.0

opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir   # output directory
echo "Output directory: $outdir"

script="python -u cnn_svgd_test.py"

outs="--output=$outdir/cnn_svgd.out --error=$outdir/cnn_svgd.err"

sbatch $opts $outs --wrap="$script"

sleep 1