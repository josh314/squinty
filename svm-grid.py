import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-n", action="store", type="int", dest="n")
parser.set_defaults(o="out.p", n=0)
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
from sklearn.grid_search import GridSearchCV

train = pd.read_csv(train_fn,header=0)

if(num_fit==0):#use all data unless specified at command line
    num_fit = len(train)
#Separate out label data and convert to numpy array
train_predictor_data = train.iloc[ :num_fit, 1:].values
train_target_data = train.iloc[ :num_fit, 0].values

#Train dat sucker
Cs = np.logspace(3, 5, 8)
#gammas = np.logspace(-8, -1, 14)
svc = svm.SVC(gamma=1e-8)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1)
clf.fit(train_predictor_data, train_target_data)

#Save model
barrel = open(model_fn, 'wb')
pickle.dump(clf, barrel)
barrel.close()

