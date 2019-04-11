#!/usr/bin/bash
module load python/3.5.0

opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir   # output directory
echo "Output directory: $outdir"

script="python -u svgd_test.py --outdir $outdir "

while IFS=' ' read -r numepochs numnns dataset
do
    outs="--output=$outdir$dataset-$numepochs-$numnns.out --error=$outdir$dataset-$numepochs-$numnns.err"
    sbatch $opts $outs --wrap="$script --num_epochs $numepochs --num_nns $numnns --dataset $dataset --outdir $outdir"
    sleep 1
done