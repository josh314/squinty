import sys
if len(sys.argv) != 2:
    print "Usage: python svm.py <pickled model output>"
    raise SystemExit(1)
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import svm

os.chdir('/Users/josh/dev/kaggle/digit-recognizer')
train = pd.read_csv('data/raw/train.csv',header=0)
test = pd.read_csv('data/raw/test.csv',header=0)

num_fit = len(train)
#Separate out label data and convert to numpy array
train_predictor_data = train.iloc[ :num_fit, 1:].values
train_target_data = train.iloc[ :num_fit, 0].values

#Train dat sucker
clf = svm.SVC(gamma=1e-8,verbose=True,shrinking=False)
clf.fit(train_predictor_data, train_target_data)

#Save model
barrel = open(sys.argv[1], 'wb')
pickle.dump(clf, barrel)
barrel.close()

