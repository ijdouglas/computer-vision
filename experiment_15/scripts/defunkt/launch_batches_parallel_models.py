#!/usr/bin/bash
# This will launch 3 subjects to modeled in parallel, at a time
# All models are allocated to gpu 1, which has enough memory to run 3 at a time
# Thus this python script is in a sequential for-loop intentionally
# The python script that this calls also request 3 CPU cores to allocate the jobs in parallel
# the python script is run_parallel_models.py
import os
subjects=['1505 1506 1507', '1508 1509 1510', '1511 1512 1513', '1514 1515 1517',
        '1518 1519 1520', '1521 1522 1523', '1524 152 1526', '1527 1528 1529']
for S_i in subjects:
    log_file="../output_logs/gpu1_parallel_batch_" + S_i.replace(' ', '-') + '.txt'
    cmd = f'nohup python run_parallel_models.py ../data/child/train ../data/child/test 1 "{S_i}" run > {log_file} &'
    os.system(cmd)
# Done
