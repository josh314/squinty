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
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

train = pd.read_csv(train_fn,header=0)

#Separate out label data and convert to numpy array
train_predictor_data = train.iloc[ :, 1:].values
train_target_data = train.iloc[ :, 0].values

#Train dat sucker
Cs = np.linspace(1,20,8)
gammas = np.linspace(1e-7, 1e-4, 8)
svc = svm.SVC()
params = dict(C=Cs,gamma=gammas)
train_size = num_fit
test_size = num_fit // 2 
cv = ShuffleSplit(len(train_predictor_data), test_size=test_size, train_size=train_size, random_state=0)
clf = GridSearchCV(estimator=svc, cv=cv, param_grid=params, n_jobs=-1)
clf.fit(train_predictor_data, train_target_data)

#Save model
barrel = open(model_fn, 'wb')
pickle.dump(clf, barrel)
barrel.close()

