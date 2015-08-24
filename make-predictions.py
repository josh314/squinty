import sys
if len(sys.argv) != 3:
    print "Usage: python k-fold-cv.py <model input> <prediction out>"
    raise SystemExit(1)
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import lda 

os.chdir('/Users/josh/dev/kaggle/digit-recognizer')

#Deserialize model
barrel = open(sys.argv[1], 'rb')
clf = pickle.load(barrel)
barrel.close()

test = pd.read_csv('data/raw/test.csv',header=0)

num_test = len(test)
#Load test data and make predictions
test_data = test.iloc[:num_test,:].values
preds = clf.predict(test_data)

#Predictions out to file
out = pd.DataFrame(zip(range(1,len(preds)+1),preds),columns=('ImageId','Label'))
out.to_csv(sys.argv[2],index=False)


#Display images and predictions for a few random obvs
num_row, num_col  = 5, 5
sample = np.random.choice(range(0,len(test_data)), num_row*num_col, replace=False)
for idx, sample_idx in enumerate(sample):
    plt.subplot(num_row, num_col, idx + 1)
    plt.axis('off')
    image = test_data[sample_idx].reshape(28,28)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%d' % preds[sample_idx])

plt.show()
