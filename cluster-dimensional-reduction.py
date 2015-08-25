################################################################
# Cluster reduction of pixel data
#
# Usage: python2 cluster-dimensional-reduction.py <infile> [options]
#
# 



#Parse command line options and arguments
#Do this before other imports so incorrect usage message appears quickly
import optparse
parser = optparse.OptionParser()

parser.add_option("-t", action="store_true", dest="t")
parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-n", action="store", type="int", dest="n")
parser.set_defaults(t=False, o="out.csv", n=28*28)
opts, args = parser.parse_args()

if(len(args) < 1):
    print "Usage: python2 cluster-dimensional-reduction.py <infile> [options]"
    raise SystemExit(1)

infile = args[0]
outfile = opts.o
is_training_data = opts.t
n_clusters = opts.n

#Other imports
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
import sklearn.cluster as cluster
import matplotlib.pyplot as plt

#Full resolution images 
fullres = pd.read_csv(infile,header=0)

num_images = len(fullres)
#If a training set, separate out label data and convert to numpy array
if(is_training_data):
    image_data = fullres.iloc[ :num_images, 1:].values
    labels = fullres.iloc[ :num_images, 0].values
else:
    image_data = fullres.iloc[ :num_images, :].values
    
#Unflatten to proper 2D shape and extract the connectivity graph
images = image_data.reshape(num_images,28,28)
connectivity = grid_to_graph(*images[0].shape)

#Do the clustering in feature space
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=n_clusters)
agglo.fit(image_data) 

#Transform the original data into reduced feature space
image_data_reduced = agglo.transform(image_data)

#Reduced data out to file
out = pd.DataFrame(image_data_reduced)
if(is_training_data):
    out.insert(0, 'label', labels)#Add back the target labels 
out.to_csv(outfile,index=False)

#
#Display some approx images to get a sense of what has been stripped out
#
#Go back to real space from eigenpixel space
# image_data_approx = agglo.inverse_transform(image_data_reduced)
# num_row, num_col  = 5, 5
# #Unflatten to 2D
# images_approx = np.reshape(image_data_approx, images.shape)
# for idx in range(0,num_row*num_col):
#     plt.subplot(num_row, num_col, idx + 1)
#     plt.axis('off')
#     plt.imshow(images_approx[idx], cmap=plt.cm.gray_r, interpolation='nearest')

# plt.show()

