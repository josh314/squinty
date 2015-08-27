import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-n", action="store", type="int", dest="n")
parser.set_defaults(o="out.p", n=1000)
opts, args = parser.parse_args()

if(len(args) < 1):
    print "Usage: python svm.py <input CSV> [options]"
    raise SystemExit(1)

train_fn = args[0]
model_fn = opts.o
num_fit = opts.n

import pickle
import pandas as pd
import numpy as np
from sklearn import svm

train = pd.read_csv(train_fn,header=0)

#Separate out label data and convert to numpy array
train_predictor_data = train.iloc[ :num_fit, 1:].values
train_target_data = train.iloc[ :num_fit, 0].values

#Estimator constructor 
clf = svm.SVC(C=3.1622776601683795,gamma=1.7782794100389229e-05,verbose=True)
#Training montage
clf.fit(train_predictor_data, train_target_data)

#Save model
barrel = open(model_fn, 'wb')
pickle.dump(clf, barrel)
barrel.close()

