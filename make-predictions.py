import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-m", action="store", type="string", dest="m")
parser.add_option("-r", action="store", type="string", dest="r")
parser.set_defaults(o="out.csv", m="", r="")
opts, args = parser.parse_args()

infile = args[0]
outfile = opts.o
model_fn = opts.m
reducer_fn = opts.r

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Deserialize reducer object for pre-processing test data
barrel = open(reducer_fn, 'rb')
reducer = pickle.load(barrel)
barrel.close()

#Load test data and make predictions
test = pd.read_csv(infile,header=0)
num_test = len(test)
test_data = test.iloc[:num_test,:].values

test_data_reduced = reducer.transform(test_data)


#Deserialize model
barrel = open(model_fn, 'rb')
clf = pickle.load(barrel)
barrel.close()

#Make predictions
preds = clf.predict(test_data_reduced)

#Predictions out to file
out = pd.DataFrame(zip(range(1,len(preds)+1),preds),columns=('ImageId','Label'))
out.to_csv(outfile,index=False)


#Display images and predictions for a few random obvs
num_row, num_col  = 5, 5
image_data_approx = reducer.inverse_transform(test_data_reduced)
sample = np.random.choice(range(0,num_test), num_row*num_col, replace=False)
for idx, sample_idx in enumerate(sample):
    plt.subplot(num_row, num_col, idx + 1)
    plt.axis('off')
    image = image_data_approx[sample_idx].reshape(28,28)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%d' % preds[sample_idx])

plt.show()
