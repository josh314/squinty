import optparse
parser = optparse.OptionParser()

parser.add_option("-t", action="store_true", dest="t")
parser.add_option("-n", action="store", type="int", dest="n")
parser.set_defaults(t=False, n=0)
opts, args = parser.parse_args()

if(len(args) < 1):
    print "Usage: python svm.py <input CSV> [options]"
    raise SystemExit(1)

infile = args[0]
is_training_set = opts.t
num_display = opts.n

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
image = image_flat.reshape(28,28)

#Add image to plot in grayscale
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
if(is_training_set):
    plt.title('%d' % label)

plt.show()
