#!/usr/bin/bash
source activate py3.6_torch1.0_pyro0.3

opts="-p batch -c 2 --mem=16000 --time=4-00:00:00 --mail-type=ALL --mail-user=$USER"

read outdir   # output directory
echo "Output directory: $outdir"

script="python -u cifar10_pytorch.py"

while IFS=' ' read -r model_num
do
    outs="--output=$outdir/$model_num.out --error=$outdir/$model_num.err"
    sbatch $opts $outs --wrap="$script --num $model_num"
    sleep 1
done
