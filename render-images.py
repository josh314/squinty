import optparse
parser = optparse.OptionParser()

parser.add_option("-t", action="store_true", dest="t")
parser.add_option("-n", action="store", type="int", dest="n")
parser.add_option("-r", action="store", type="string", dest="r")
parser.set_defaults(t=False, n=0, r="")
opts, args = parser.parse_args()

if(len(args) < 1):
    print "Usage: python svm.py <input CSV> [options]"
    raise SystemExit(1)

infile = args[0]
is_training_set = opts.t
num_display = opts.n
reducer_fn = opts.r

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv(infile, header=0)

#Label if applicable
label = ""
if(is_training_set):
    label = train.loc[num_display,'label']


#Get the pixel data for one of the images.
#column 1 is the training label if applicable, so skip that 
#Expand the flat array into 2-dimensional data. Images are 28x28 pixels
skip = 1 if is_training_set else 0
image_flat = train.iloc[num_display,skip:].values

reducer = None
if(reducer_fn!=""):
    #Deserialize reducer object for pre-processing test data
    barrel = open(reducer_fn, 'rb')
    reducer = pickle.load(barrel)
    barrel.close()
    image_reduced = reducer.transform(image_flat)    
    image_flat = reducer.inverse_transform(image_reduced)

image = image_flat.reshape(28,28)

#Add image to plot in grayscale
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
if(is_training_set):
    plt.title('%d' % label)

plt.show()
