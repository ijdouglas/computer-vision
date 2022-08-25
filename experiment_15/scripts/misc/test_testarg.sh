#!/usr/bin/bash
param1='1 2 3'
param2='3 4'
nohup python testarg.py $param1 > out1.txt
nohup python testarg.py $param2 > out2.txt
