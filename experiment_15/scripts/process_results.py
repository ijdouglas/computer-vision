import os
import re
import pandas as pd
import argparse
# Setup args
parser = argparse.ArgumentParser(description="Concatenate all the results across either children or parents")
parser.add_argument("--child_or_parent", 
                    help="supply either `child` or `parent` to process either child or parent data",
                    default = None)
args= parser.parse_args()
testf_path = '../data/exp15/' + args.child_or_parent + '/' + args.child_or_parent + '_test_'
resf_path = '../results/exp15/' + args.child_or_parent + '/' + args.child_or_parent + '_test_'
outpath = '../results/exp15/' + args.child_or_parent + '/all_results.csv'
out = []
for i in range(1, 54):
    testf = testf_path + str(i) + '.csv'
    resf = resf_path + str(i) + '_results.csv'
    t = pd.read_csv(testf, index_col=None)
    # extract the image number to uniquely identify each row and merge by
    tsearch = []
    for values in t['path']:
        tsearch.append(re.search('img_[0-9]+.jpg', values).group())
    t['path'] = tsearch
    t = t.rename(columns={"path": "img"})
    # repeat for results
    r = pd.read_csv(resf, index_col=None)
    rsearch = []
    for values in r['img']:
        rsearch.append(re.search('img_[0-9]+.jpg', values).group())
    r['img'] = rsearch
    out.append(pd.merge(t, r, on = 'img'))
pd.concat(out, axis = 0).to_csv(outpath, index=False)
