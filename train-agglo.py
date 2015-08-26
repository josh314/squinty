################################################################
# Cluster reduction of pixel data
#
# Usage: python2 train-agglo.py <training data> [options]
#
# Builds the a data transformer (FeatureAgglomeration clusting object) which
# reduces problem dimension

#Parse command line options and arguments
#Do this before other imports so incorrect usage message appears quickly
import optparse
parser = optparse.OptionParser()

parser.add_option("-o", action="store", type="string", dest="o")
parser.add_option("-n", action="store", type="int", dest="n")
parser.add_option("-r", action="store", type="string", dest="r")
parser.set_defaults(o="out.csv", n=14*14, r="reducer.p")
opts, args = parser.parse_args()

if(len(args) < 1):
    print "Usage: python2 train-agglo.py <training data> [options]"
    raise SystemExit(1)

infile = args[0]
outfile = opts.o
n_clusters = opts.n
reducer_fn = opts.r

#Other imports
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
import sklearn.cluster as cluster


#Full resolution images 
fullres = pd.read_csv(infile,header=0)

num_images = len(fullres)
image_data = fullres.iloc[ :num_images, 1:].values
labels = fullres.iloc[ :num_images, 0].values
    
#Unflatten to proper 2D shape and extract the connectivity graph
images = image_data.reshape(num_images,28,28)
connectivity = grid_to_graph(*images[0].shape)

#Do the clustering in feature space
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=n_clusters)
agglo.fit(image_data)

# Save agglo for future use in workflow.
# Must use this same agglo for test data
barrel=open(reducer_fn,'wb')
pickle.dump(agglo,barrel)
barrel.close()

#Transform the original data into reduced feature space
image_data_reduced = agglo.transform(image_data)

#Reduced data out to file
out = pd.DataFrame(image_data_reduced)
out.insert(0, 'label', labels)#Add back the target labels 
out.to_csv(outfile,index=False)

### Uncomment below to see some demo approx images

# import matplotlib.pyplot as plt
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

