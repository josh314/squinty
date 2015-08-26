This repo contains code for solving a basic machine vision proble, specifically the [Kaggle digit recognizer competition](https://www.kaggle.com/c/digit-recognizer). The goal is to correctly label images with the Arabic numerals they picture. The training data is a set of 42,000 labeled images, each with 784 pixels (28 pixels by 28 pixels square image). The test data consists of 28,000 such images. All data is in CSV format.

The solution scenario was to use a feature agglomeration to reduce the problem dimension. A SVC is then trained on this reduced training data. Predictions are made by performing the same reduction on the test data before applying the trained SVC. Hyperparameter selection for the SVC is done by a combination of cross validated grid search and manual selection.

The code consists of a number of short command line scripts. The workflow digests the training data in a series of scripts that build models and transformation objects and saves them to file via pickle. The predictions scripts consume these pickled objects and the test data to produce the prediction file.

Example Workflow
================
In detail, an example workflow as follows.

`train-agglo.py data/raw/train.cvs -n49 -r agglo-49.p -o data/build/train-49.csv`

This trains and pickles a reducer object (in this case `sklearn.featureAgglomeration` with 49 clusters). The transformed training data is saved in the `train-49.csv` file. Use this transformed training data to train a model

`svm-grid.py data/build/train-49.csv -o svm-49-model.p -n5000`

The `-n` flag sets the number of training points per CV-fold. The trained model is output to `svm-49-model.p`. Finally, predictions for test data can be obtained

`make-predictions.py data/raw/test.csv -o predictions.csv -m svm-49-model.p -r agglo.p`

using flags to indicate the output predictions file, and input model and reducer objects.

Image rendering code
=====================
There is a little image rendering code in this repo as well.
 
- `render-images.py` will display a single image and its label from either the full resolution training set or test set. Use the `-t` flag for the training set 
- After performing feature clustering, a block at the end of `train-agglo.py` displays 25 random reduced training images as a demo
- After calculating the test set predictions the `make-predictions.py` script displays 25 random reduced test images and labeled with their predictions. 
