#!/usr/bin/python
import os
import sys
import datetime
import joblib
from joblib import Parallel, delayed
# Need to add arguments for directory to train and test, and possibly GPU ID
# and number of cores, and the list of subjects
args = sys.argv
# Note args[0] is the name of the script
train_root = args[1]
test_root = args[2]
gpu_id = args[3]
#subs = args[4].split(' ') # must be '1501 1502' for example
run = args[5]
# EDIT: looping through a list in batches instead
subjects=['1505 1506 1507', '1508 1509 1510', '1511 1512 1513', '1514 1515 1517',
        '1518 1519 1520', '1521 1522 1523', '1524 152 1526', '1527 1528 1529']
# NOTE THERE IS A TYPO FOR 1525: they are included in the second batch (gpu0)
#if run == 'run':
#   do the whole script
#   else:
#     print an explanation of each argument, 
#     and that the last arg must be 'run' to actually run

#paths = os.listdir('../data/child/train') # relative to scripts folder
#N = int(len(paths))
#subs = [1501, 1503]; N = len(subs)
def launch_model(subject_id):
    trainf = os.path.join(train_root, str(subject_id) + '_train.csv')
    testf = os.path.join(test_root, str(subject_id) + '_test.csv')
    maincall = 'python train.py'
    rootdir = ' --root_dir /data/drives/multiwork'
    tr = ' --train_set ' + trainf
    te = ' --test_set ' + testf
    gpu = ' --gpu ' + str(gpu_id)
    model = ' --model resnet152'
    freeze = ' --freeze_backbone True'
    epochs = ' --epochs 30'
    lr = ' --lr .0005'
    batch = ' --batch 64'
    cmd = maincall + rootdir + tr + te + gpu + model + epochs + freeze + lr #+ batch
    print('\n Running call: ' + str(cmd) + '\n')
    print(f'Current Time: {datetime.datetime.now()}')
    os.system(cmd)
    print("\n")
    print(f'Finished running {subject_id} at {datetime.datetime.now()}')
for subs in subjects:
    #Run for all subjects in `subs`
    print('Running script: ' + args[0] + ' with subjects: ' + subs) 
    if __name__ == '__main__':
        Parallel(n_jobs=len(subs))(delayed(launch_model)(s) for s in subs.split(' '))
#done
