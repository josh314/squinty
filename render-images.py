import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.chdir('/Users/josh/dev/kaggle/digit-recognizer')
train = pd.read_csv('data/raw/train.csv', header=0)

#Which image do we want to view? Doesn't really matter, just pick one
image_idx = 2300
label = train.loc[image_idx,'label']

#Print the label for verification purpose
print("Image %d has been labeled as a numeral %d\n" % (image_idx, label))

#Get the pixel data for one of the images.
#column 1 is the training label, so skip that 
image_flat = train.iloc[image_idx,1:].values

#Expand the flat array into 2-dimensional data. Images are 28x28 pixels
#image = np.reshape(image_flat, (28,28)) #Alternate equivalent static command
image = image_flat.reshape(28,28)

#Add image to plot in grayscale
plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
