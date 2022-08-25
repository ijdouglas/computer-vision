#!/usr/bin/python
# SHould be run with a call such as:
# `nohup python THIS_SCRIPT.py path/to/train path/to/train 0 'run'
import os
import sys
import datetime
import joblib
from glob import glob
from joblib import Parallel, delayed
args = sys.argv
# Note args[0] is the name of the script
print(f"Running script: {args[0]} at {datetime.datetime.now()}")
if len(args) == 11:
    train_root = args[1] # directory with all train sets
    test_root = args[2] # directory with all test set
    gpu_id = args[3] # gpu to use (0 or 1)
    label = args[4] # variable name of the response labels
    normal = args[5] # normalization to use 'mnist' or 'imagenet'
    nclass = args[6] # number of classes in label/response
    res_dir = args[7] # where to save out the results (testing accuracy)
    n_epochs = args[8]
    models_dir = args[9] # where to save the trained model for each subject
    run = args[10] # must be 'run' or else a dry run occurs
else:
    raise ValueError('Script requires the following ordered/positional arguments:\n1. directory with training sets\n2. directory with testing sets\n3. gpu id\n4. the variable name of the response labels. \n5. type of normalization to use (passed to --label_key in train.py).\n6. Number of classes in the response variable/labels.\n7. The directory where to save the results csv files for all subjects. \n8. The number of epochs to run. \n9. The directory in which to save all of the models `[subject]_model.pt` and summary stats `[subject]_info.pkl`. \n10. "run" to run or anything else to dry run/print help')
print(f"training directory: {args[1]}")
print(f"Testing directory: {args[2]}")
print(f"Using GPU with GPU ID: {args[3]}")
print(f"Modeling variable: {args[4]}")
print(f"Number of epochs: {args[8]}")
print(f"Using normalization: {args[5]}")
print(f"Number of classes in response: {args[6]}")
print(f"Saving trained models in subject-specific subdirectories of: {args[9]}")
print(f"Saving results to: {res_dir}")
# EDIT: looping through a list in batches instead
subjects = ['1201 1202 1203 1204 1205',
            '1208 1209 1210 1211 1212',
            '1213 1214 1215 1216 1217',
            '1218 1219 1220 1221 1222']
print(f"Running the following subject batches: {subjects}")
def launch_model(subject_id):
    trainf = glob(os.path.join(train_root, str(subject_id) + '*csv'))[0] # should be 1 match!
    testf = glob(os.path.join(test_root, str(subject_id) + '*csv'))[0]
#    trainf = os.path.join(train_root, str(subject_id) + '_train.csv')
#    testf = os.path.join(test_root, str(subject_id) + '_test.csv')
    maincall = 'python train.py'
    rootdir = ' --root_dir /data/drives/multiwork'
    tr = ' --train_set ' + trainf
    te = ' --test_set ' + testf
    chkpt = ' --chkpt_dir ' + models_dir
    gpu = ' --gpu ' + str(gpu_id)
    model = ' --model resnet152'
    freeze = ' --freeze_backbone True'
    epochs = ' --epochs ' + n_epochs
    lr = ' --lr .0005'
    batch = ' --batch 64'
    norm = ' --normalization ' + normal
    n = ' --num_classes ' + nclass
    L = ' --label_key ' + label
    res = ' --results_dir ' + res_dir
    cmd = maincall + rootdir + tr + te + chkpt + gpu + model + epochs + freeze + lr + batch + norm + n + L + res
    print('\n Running call: ' + str(cmd) + '\n')
    print(f'Starting Time: {datetime.datetime.now()}')
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
    print('This is a dry run. To execute the script, provide as the 10th argument "run". Arguments (in order; required):\n1. directory with training sets\n2. directory with testing sets\n3. gpu id\n4. the variable name of the response labels. \n5. type of normalization to use (passed to --label_key in train.py).\n6. Number of classes in the response variable/labels.\n7. The directory where to save the results csv files for all subjects. \n8. The number of epochs to run. \n9. The directory in which to save all of the models `[subject]_model.pt` and summary stats `[subject]_info.pkl`. \n10. "run" to run or anything else to dry run/print help')
#done
