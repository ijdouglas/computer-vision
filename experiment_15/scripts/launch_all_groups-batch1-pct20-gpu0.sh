#!/usr/bin/bash
parameters='../data/child/pct20_train ../data/child/test 0 run'
#nohup python launch_pct20_batch1_group-1-of-6.py > child_batch1-group-1_gpu0.txt
nohup python launch_pct20_batch1_group-2-of-6.py $parameters > ../output_logs/pct20/child_batch1-group-2_gpu0.txt
nohup python launch_pct20_batch1_group-3-of-6.py $parameters > ../output_logs/pct20/child_batch1-group-3_gpu0.txt
nohup python launch_pct20_batch1_group-4-of-6.py $parameters > ../output_logs/pct20/child_batch1-group-4_gpu0.txt
nohup python launch_pct20_batch1_group-5-of-6.py $parameters > ../output_logs/pct20/child_batch1-group-5_gpu0.txt
nohup python launch_pct20_batch1_group-6-of-6.py $parameters > ../output_logs/pct20/child_batch1-group-6_gpu0.txt
