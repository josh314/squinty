This repo contains code for solving a basic machine vision proble, specifically the [Kaggle digit recognizer competition](https://www.kaggle.com/c/digit-recognizer). The goal is to correctly label images with the Arabic numerals they picture. The training data is a set of 42,000 labeled images, each with 784 pixels (28 pixels by 28 pixels square image). The test data consists of 28,000 such images. All data is in CSV format.

The code consists of a number of short command line scripts. The workflow digests the training data in a series of scripts that build models and transformation objects and saves them to file via pickle. The predictions scripts consume these pickled objects and the test data to produce the prediction file. 