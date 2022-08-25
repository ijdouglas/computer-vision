import os
import re
import pandas as pd
import glob
# THIS IS THE PARENT VERSION
# To speed up training (for the purposes of ML final project only)
# sample some percent of the training frames (will still results in thousands of frames).
# write this out as tmp_[original file path.csv] and when calling train.py,
# point the train_set argument to this new subset file.
# Note, results are written out to a file named something based on test_set, so
# this doesn't have any downstream consequences.
# Note 2: when creating the new sample, sample an equal number of JA and not JA
#paths = os.listdir('../data/exp15/parent') # relative to scripts folder
paths = glob.glob('../data/exp15/parent/parent_train*')
N = int(len(paths))
for i in range(N):
    n = i + 1 # because range starts at 0
    trainf = '../data/exp15/parent/parent_train_' + str(n) + '.csv'
    tmp_trainf = '../data/exp15/parent/tmp_parent_train_' + str(n) + '.csv'
    tmp = pd.read_csv(trainf, index_col=None)
    # Figure out how many of each case to sample based on the number of each case
    N_min = min(tmp.groupby('joint_attention').apply(lambda x: len(x)))
    N_frac = int(.2*N_min)
    # Sample N_frac rows from each group (groups are joint-attention and not-joint-attention:
    tmp = tmp.groupby('joint_attention', group_keys=False).apply(lambda x: x.sample(N_frac)).reset_index(drop=True)
    tmp.to_csv(tmp_trainf, index=False) # save out the downsampled data
    testf = '../data/exp15/parent/parent_test_' + str(n) + '.csv'
    # Compose the call (USING tmp_trainf)
    maincall = 'python train.py' # we're in the scripts folder already 
    rootdir = ' --root_dir /data/drives/multiwork'
    tr = ' --train_set ' + tmp_trainf # !!!
    te = ' --test_set ' + testf
    gpu = ' --gpu 1'
    model = ' --model resnet152'
    freeze = ' --freeze_backbone True'
    epochs = ' --epochs 5'
    lr = ' --lr .0005'
    batch = ' --batch 128'
    cmd = maincall + rootdir + tr + te + gpu + model + epochs + freeze + lr + batch
    os.system(cmd)
    print('\n Running call: ' + str(cmd))
# Done
