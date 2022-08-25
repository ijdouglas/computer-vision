#!/usr/bin/bash
# This runs with GNU bash, version 5.0.17(1)-release (x86_64-pc-linux-gnu)
# using: `bash this_script.sh`
# It will launch 3 subjects to modeled in parallel, at a time
# All models are allocated to gpu 1, which has enough memory to run 3 at a time
# The python script that this calls also request 3 CPU cores to allocate the jobs in parallel
# the python script is run_parallel_models.py
for s in '1505 1506 1507' '1508 1509 1510' '1511 1512 1513' '1514 1515 1517' '1518 1519 1520' '1521 1522 1523' '1524 152 1526' '1527 1528 1529'; do
	out_file="../output_logs/gpu1_parallel_batch_${s// /-}.txt"
	nohup python run_parallel_models.py "../data/child/train" "../data/child/test" 1 $s 'run' > $out_file &
done
