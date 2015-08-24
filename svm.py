import os
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

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

#Load test data and make predictions
test_data = test.iloc[:,:].values
preds = clf.predict(test_data)

#How good was the fit on the training data?
clf.score(train_predictor_data,train_target_data)

#Predictions out to file
out = pd.DataFrame(zip(range(1,len(preds)+1),preds),columns=('ImageId','Label'))
out.to_csv('res.svm.csv',index=False)


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
