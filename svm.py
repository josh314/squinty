import sys
if len(sys.argv) != 3:
    print "Usage: python svm.py <training data> <pickled model output>"
    raise SystemExit(1)

train_fn, model_fn = sys.argv[1], sys.argv[2]

import pickle
import pandas as pd
import numpy as np
from sklearn import svm

train = pd.read_csv(train_fn,header=0)

num_fit = len(train)
#Separate out label data and convert to numpy array
train_predictor_data = train.iloc[ :num_fit, 1:].values
train_target_data = train.iloc[ :num_fit, 0].values

#Estimator constructor 
clf = svm.SVC(gamma=1e-8,verbose=True,shrinking=False)
#Training montage
clf.fit(train_predictor_data, train_target_data)

#Save model
barrel = open(model_fn, 'wb')
pickle.dump(clf, barrel)
barrel.close()

