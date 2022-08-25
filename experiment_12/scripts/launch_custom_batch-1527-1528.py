#!/usr/bin/python
# SHould be run with a call such as:
# `nohup python THIS_SCRIPT.py path/to/train path/to/train 0 'run'
import os
import sys
import datetime
import joblib
from joblib import Parallel, delayed
args = sys.argv
# Note args[0] is the name of the script
print(f"Running script: {args[0]} at {datetime.datetime.now()}")
if len(args) == 8:
    train_root = args[1] # directory with all train sets
    test_root = args[2] # directory with all test set
    gpu_id = args[3] # gpu to use (0 or 1)
    label = args[4] # variable name of the response labels
    normal = args[5] # normalization to use 'mnist' or 'imagenet'
    nclass = args[6] # number of classes in label/response
    run = args[7] # must be 'run' or else a dry run occurs
else:
    raise ValueError('Script requires the following ordered/positional arguments:\n1. directory with training sets\n2. directory with testing sets\n3. gpu id\n4. the variable name of the response labels. \n5. type of normalization to use (passed to --label_key in train.py).\n6. Number of classes in the response variable/labels. \n7. "run" to run or anything else to dry run/print help')
print(f"training directory: {args[1]}")
print(f"Testing directory: {args[2]}")
print(f"Using GPU with GPU ID: {args[3]}")
print(f"Modeling variable: {args[4]}")
print(f"Using normalization: {args[5]}")
print(f"Number of classes in response: {args[6]}")
# EDIT: looping through a list in batches instead
subjects=['1527 1528']
print(f"Running the following subject batches: {subjects}")
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
    norm = ' --normalization ' + normal
    n = ' --num_classes ' + nclass
    L = ' --label_key ' + label
    cmd = maincall + rootdir + tr + te + gpu + model + epochs + freeze + lr + batch + norm + n + L
    print('\n Running call: ' + str(cmd) + '\n')
    print(f'Current Time: {datetime.datetime.now()}')
    os.system(cmd)
    print("\n")
    print(f'Finished running {subject_id} at {datetime.datetime.now()}')

if run == 'run':
    for subs in subjects:
        #Run for all subjects in `subs`
        print('Running train.py with subject batch: ' + subs)
        if __name__ == '__main__':
            Parallel(n_jobs=len(subs))(delayed(launch_model)(s) for s in subs.split(' '))
else:
    print('This is a dry run. To execute the script provide "run" as the fourth positional argument. Arguments:\n1. directory with training sets\n2. directory with testing sets\n3.gpu id\n4. "run" to run or anything else to dry run/print help')
#donei
